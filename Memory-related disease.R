############-------------XGboost-----------------##############
# 加载必要的包
library(data.table)
library(xgboost)
library(rBayesianOptimization)
library(readxl)
library(caret)
library(pROC)
library(ggplot2)

# 1. 数据准备
train_data <- read_excel("C:/Users/16039/Desktop/单病外部验证/记忆问题训练集.xlsx")
test_data <- read_excel("C:/Users/16039/Desktop/单病外部验证/记忆问题验证集.xlsx")

# 处理缺失值
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# 确保测试数据变量顺序与训练数据完全一致
test_data <- test_data[, colnames(train_data)]

# 验证顺序是否匹配
if(identical(colnames(train_data), colnames(test_data))) {
  print("变量顺序已完全匹配！")
} else {
  print("警告：变量顺序仍不匹配！")
  cat("训练数据特征顺序:\n", colnames(train_data))
  cat("\n测试数据特征顺序:\n", colnames(test_data))
}

# 创建统一的编码规则（基于训练集）
dummy_model <- dummyVars(~ ., 
                         data = train_data[,3:8], 
                         fullRank = FALSE)

# 应用到训练集和测试集
train_encoded <- predict(dummy_model, train_data[,3:8])
test_encoded <- predict(dummy_model, test_data[,3:8])

# 合并编码后的部分与其他列
train_final <- cbind(
  train_data[, -c(3:8)],  # 保留除2-11列外的其他列
  train_encoded
)

test_final <- cbind(
  test_data[, -c(3:8)],   # 保留除2-11列外的其他列
  test_encoded
)

# 确保测试集包含训练集的所有列
missing_cols <- setdiff(colnames(train_final), colnames(test_final))
if(length(missing_cols) > 0) {
  missing_data <- matrix(0, nrow = nrow(test_final), ncol = length(missing_cols))
  colnames(missing_data) <- missing_cols
  test_final <- cbind(test_final, missing_data)
}

# 按训练集的列顺序排列测试集
test_final <- test_final[, colnames(train_final)]

# 转换为矩阵
train_matrix <- as.matrix(train_final)
test_matrix <- as.matrix(test_final)

# 验证结果
cat("训练集维度:", dim(train_matrix), "\n")
cat("测试集维度:", dim(test_matrix), "\n")
cat("列名是否一致:", identical(colnames(train_matrix), colnames(test_matrix)))

# 创建DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(train_matrix[, -4]), label = train_data$memory_problem)
dtest <- xgb.DMatrix(data = as.matrix(test_matrix[, -4]), label = test_data$memory_problem)

# 2. 第一阶段：大范围随机搜索
cat("\n===== 开始第一阶段：大范围随机搜索 =====\n")

# 定义随机搜索函数
random_search_xgb <- function(n_iter = 50) {
  best_auc <- 0
  best_params <- NULL
  best_iteration <- 0
  
  # 大范围参数空间
  search_space <- list(
    max_depth = c(2L, 15L),
    min_child_weight = c(1L, 100L),
    colsample_bytree = c(0.3, 1.0),
    subsample = c(0.5, 1.0),
    gamma = c(0, 10),
    alpha = c(0, 20),
    eta = c(0.005, 0.5),
    nrounds = c(10L, 1000L),
    lambda = c(0, 50)
  )
  
  for(i in 1:n_iter) {
    cat(sprintf("\n随机搜索迭代 %d/%d", i, n_iter))
    
    # 随机采样参数
    params <- list(
      max_depth = sample(search_space$max_depth[1]:search_space$max_depth[2], 1),
      min_child_weight = runif(1, search_space$min_child_weight[1], search_space$min_child_weight[2]),
      colsample_bytree = runif(1, search_space$colsample_bytree[1], search_space$colsample_bytree[2]),
      subsample = runif(1, search_space$subsample[1], search_space$subsample[2]),
      gamma = runif(1, search_space$gamma[1], search_space$gamma[2]),
      alpha = runif(1, search_space$alpha[1], search_space$alpha[2]),
      eta = runif(1, search_space$eta[1], search_space$eta[2]),
      lambda = runif(1, search_space$lambda[1], search_space$lambda[2]),
      nrounds = sample(search_space$nrounds[1]:search_space$nrounds[2], 1)
    )
    
    # 设置模型参数
    model_params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = params$max_depth,
      min_child_weight = params$min_child_weight,
      colsample_bytree = params$colsample_bytree,
      subsample = params$subsample,
      gamma = params$gamma,
      alpha = params$alpha,
      eta = params$eta,
      lambda = params$lambda
    )
    
    # 带早停的交叉验证
    cv <- xgb.cv(
      params = model_params,
      data = dtrain,
      nrounds = params$nrounds,
      nfold = 5,
      early_stopping_rounds = 20,
      maximize = TRUE,
      verbose = 0,
      stratified = TRUE
    )
    
    # 获取最佳AUC
    best_auc_score <- max(cv$evaluation_log$test_auc_mean)
    best_iter <- cv$best_iteration
    
    # 更新最佳参数
    if(best_auc_score > best_auc) {
      best_auc <- best_auc_score
      best_params <- params
      best_params$best_iter <- best_iter
      cat(sprintf(" - 发现新最佳AUC: %.4f", best_auc))
    }
  }
  
  return(list(best_auc = best_auc, best_params = best_params))
}

# 执行随机搜索 (100次迭代)
set.seed(123)
random_results <- random_search_xgb(n_iter = 100)

# 打印随机搜索结果
cat("\n\n===== 随机搜索完成 =====")
cat("\n最佳验证AUC:", random_results$best_auc)
cat("\n最佳参数组合:\n")
print(random_results$best_params)

# 3. 第二阶段：基于随机搜索结果的贝叶斯优化
cat("\n\n===== 开始第二阶段：贝叶斯优化 =====\n")

# 根据随机搜索结果确定精细搜索范围
refined_params <- list(
  max_depth = c(max(3L, random_results$best_params$max_depth - 2L), 
                min(20L, random_results$best_params$max_depth + 2L)),
  
  min_child_weight = c(max(1L, random_results$best_params$min_child_weight * 0.7), 
                       min(100L, random_results$best_params$min_child_weight * 1.3)),
  
  colsample_bytree = c(max(0.3, random_results$best_params$colsample_bytree - 0.15), 
                       min(1.0, random_results$best_params$colsample_bytree + 0.15)),
  
  subsample = c(max(0.5, random_results$best_params$subsample - 0.15), 
                min(1.0, random_results$best_params$subsample + 0.15)),
  
  gamma = c(max(0, random_results$best_params$gamma - 5), 
            min(30, random_results$best_params$gamma + 5)),
  
  alpha = c(max(0, random_results$best_params$alpha - 5), 
            min(30, random_results$best_params$alpha + 5)),
  
  eta = c(max(0.001, random_results$best_params$eta * 0.5), 
          min(0.5, random_results$best_params$eta * 1.5)),
  
  nrounds = c(max(50L, random_results$best_params$best_iter - 50L), 
              min(1000L, random_results$best_params$best_iter + 200L)),
  
  lambda = c(max(0, random_results$best_params$lambda - 10), 
             min(50, random_results$best_params$lambda + 10))
)

# 打印精炼后的参数范围
cat("\n精炼后的贝叶斯优化搜索范围:\n")
print(refined_params)

# 定义贝叶斯优化的目标函数
xgb_cv_bayes <- function(max_depth, min_child_weight, colsample_bytree, gamma, alpha, eta, nrounds, lambda, subsample) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    alpha = alpha,
    eta = eta,
    lambda = lambda,
    subsample = subsample
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = as.integer(nrounds),
    nfold = 5,
    early_stopping_rounds = 20,
    maximize = TRUE,
    verbose = 0,
    stratified = TRUE
  )
  
  # 返回最佳迭代的AUC
  return(list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration]))
}

# 运行贝叶斯优化
set.seed(42)
xgb_bo <- BayesianOptimization(
  FUN = xgb_cv_bayes,
  bounds = refined_params,
  init_points = 15,
  n_iter = 30,
  acq = "ucb",
  kappa = 2.576,
  verbose = TRUE
)

# 提取贝叶斯优化结果
cat("\n\n===== 贝叶斯优化完成 =====")
cat("\n最佳AUC:", xgb_bo$Best_Value)
cat("\n最佳参数组合:\n")
print(xgb_bo$Best_Par)


# Best Parameters Found: 
#   Round = 40	max_depth = 4.0000	min_child_weight = 25.09737	colsample_bytree = 0.6406049	subsample = 0.8475186	gamma = 2.220446e-16	alpha = 3.732319	eta = 0.0627916	nrounds = 213.0000	lambda = 1.716461	Value = 0.7243834 



# 4. 使用最佳参数训练最终模型
best_bayes_params <- xgb_bo$Best_Par

final_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = as.integer(4),
  min_child_weight =25.09737 ,
  colsample_bytree = 0.6406049,
  subsample = 0.8475186,
  gamma = 2.220446e-16,
  alpha = 3.732319,
  eta = 0.0627916,
  lambda = 1.716461
)



# 训练模型
set.seed(123)
xgb_final <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 213,
  maximize = TRUE,
  print_every_n = 10
)


# 5. 模型评估
# 预测测试集
test_pred <- predict(xgb_final, dtest)

# 计算训练集预测和AUC
train_pred <- predict(xgb_final, dtrain)
roc_train <- roc(train_data$memory_problem, train_pred)
auc_train <- auc(roc_train)
ci_train <- ci.auc(roc_train)

# 计算测试集AUC
roc_test <- roc(test_data$memory_problem, test_pred)
auc_test <- auc(roc_test)
ci_test <- ci.auc(roc_test)

# 打印测试结果
cat("\n\n===== 最终模型测试结果 =====")
cat("\n训练集AUC:", round(auc_train, 4), "95% CI:", round(ci_train[1], 4), "-", round(ci_train[3], 4))
cat("\n测试集AUC:", round(auc_test, 4), "95% CI:", round(ci_test[1], 4), "-", round(ci_test[3], 4))

# 6. 可视化结果
# 变量重要性
importance_matrix <- xgb.importance(model = xgb_final)

# 绘制变量重要性图
imp_plot <- ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Variable", y = "Importance (Gain)", 
       title = "XGBoost Feature Importance") +
  theme_minimal()

print(imp_plot)

# 准备ROC曲线数据
roc_data <- rbind(
  data.frame(
    Specificity = roc_train$specificities,
    Sensitivity = roc_train$sensitivities,
    Dataset = paste0("训练集: AUC = ", round(auc_train, 3), 
                     " (95% CI: ", round(ci_train[1], 3), "-", round(ci_train[3], 3), ")")
  ),
  data.frame(
    Specificity = roc_test$specificities,
    Sensitivity = roc_test$sensitivities,
    Dataset = paste0("测试集: AUC = ", round(auc_test, 3), 
                     " (95% CI: ", round(ci_test[1], 3), "-", round(ci_test[3], 3), ")")
  )
)

# 绘制ROC曲线
roc_plot_memory <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Dataset)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(x = "1 - 特异度", 
       y = "灵敏度"
      # title = "ROC Curves for Memory Problem Prediction"
       ) +
  scale_color_manual(values = c("#377eb8", "#e41a1c")) + # 蓝色为训练集，红色为测试集
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

print(roc_plot_memory)


