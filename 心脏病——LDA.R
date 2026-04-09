# ############-------------线性判别分析（LDA）-----------------##############
# 加载必要的包
library(MASS)        # 提供 lda 函数
library(caret)       # 用于混淆矩阵和评估
library(pROC)        # 绘制 ROC 曲线
library(ggplot2)     # 可选绘图
library(readxl)
# 假设你已经运行了之前代码中的数据读取和缺失值处理部分：
train_data <- read_excel("C:/Users/16039/Desktop/单病外部验证/心脏病训练集.xlsx")
test_data <- read_excel("C:/Users/16039/Desktop/单病外部验证/心脏病验证集.xlsx")

# 处理缺失值
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# 确保测试数据变量顺序与训练数据完全一致
test_data <- test_data[, colnames(train_data)]

library(candisc)
X <- train_data[, 1:14]
mlm_model <- lm(as.matrix(X) ~ heart_disease, data = train_data)
candisc_result <- candisc(mlm_model, term = "heart_disease")
print(candisc_result)
summary(candisc_result)   # 包含 Wilks' Lambda 检验详细表格

canonical_cor <- candisc_result$canrsq
# 计算典型相关系数
canonical_cor <- sqrt(candisc_result$canrsq)
print(canonical_cor)  


# 1. 数据预处理（适用于 LDA）
# ------------------------------------------------------------
# 确保因变量为因子（二分类或多分类）
train_data$heart_disease <- as.factor(train_data$heart_disease)
test_data$heart_disease  <- as.factor(test_data$heart_disease)

# 将自变量中的字符型变量转为因子，数值型保持原样
# 定义分类变量的列索引（3-14）
cat_cols <- 3:14
# 将分类变量统一转换为因子
for (col in cat_cols) {
  train_data[[col]] <- as.factor(train_data[[col]])
  test_data[[col]]  <- as.factor(test_data[[col]])
}
# 检查因子变量的水平在训练集和测试集是否一致
# 对于因子变量，测试集中不能出现训练集未出现过的新水平
for (col in cat_cols) {
  test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
}

# 2. 训练 LDA 模型
# ------------------------------------------------------------
# 使用所有自变量预测 heart_disease
lda_model <- lda(heart_disease ~ ., data = train_data)

# 提取非标准化系数（LD1）
raw_coef <- coef(lda_model)  # 或 lda_model$scaling



# 3. 预测
# ------------------------------------------------------------
# 训练集预测
train_pred <- predict(lda_model)  # 默认使用训练数据
# 测试集预测
test_pred  <- predict(lda_model, newdata = test_data)

# 预测结果包含三部分：
# class: 预测的类别
# posterior: 后验概率矩阵
# x: 判别得分（线性判别函数值）

# 4. 模型评估
# ------------------------------------------------------------
# 混淆矩阵（训练集）
cat("\n========== 训练集混淆矩阵 ==========\n")
train_cm <- confusionMatrix(train_pred$class, train_data$heart_disease)
print(train_cm)

# 混淆矩阵（测试集）
cat("\n========== 测试集混淆矩阵 ==========\n")
test_cm <- confusionMatrix(test_pred$class, test_data$heart_disease)
print(test_cm)

# 提取测试集准确率
test_accuracy <- test_cm$overall["Accuracy"]
cat("\n测试集准确率:", round(test_accuracy, 4), "\n")

roc_train <- roc(train_data$heart_disease, train_pred$posterior[, 2])
cat("\n训练集 AUC:", round(auc(roc_train), 4), "\n")

roc_test <- roc(test_data$heart_disease, test_pred$posterior[, 2])
cat("\n测试集 AUC:", round(auc(roc_test), 4), "\n")

# 计算 AUC 的 95% 置信区间（默认方法 = "delong"）
ci_train <- ci.auc(roc_train)
ci_test  <- ci.auc(roc_test)
# 准备ROC曲线数据
roc_data <- rbind(
  data.frame(
    Specificity = roc_train$specificities,
    Sensitivity = roc_train$sensitivities,
    Dataset = paste0("训练集: AUC = ", round(auc(roc_train), 3), 
                     " (95% CI: ", round(ci_train[1], 3), "-", round(ci_train[3], 3), ")")
  ),
  data.frame(
    Specificity = roc_test$specificities,
    Sensitivity = roc_test$sensitivities,
    Dataset = paste0("测试集: AUC = ", round(auc(roc_test), 3), 
                     " (95% CI: ", round(ci_test[1], 3), "-", round(ci_test[3], 3), ")")
  )
)

# 绘制ROC曲线
roc_plot_heart <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Dataset)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(x = "1 - 特异度", 
       y = "灵敏度"
  ) +
  scale_color_manual(values = c("#377eb8", "#e41a1c")) + # 蓝色为训练集，红色为测试集
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

print(roc_plot_heart)
