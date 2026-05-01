# ==============================================================================
# 辅助函数 1 (造轮子): 寻找向量中目标值的索引
# 对应 C++: std::find 或基础的 for 循环
# ==============================================================================
matrix_freq_helper_find_index <- function(vec, target) {
  out_idx <- -1
  for (i in base::seq_along(vec)) {
    if (vec[i] == target) {
      out_idx <- i
      break
    }
  }
  return(out_idx)
}

# ==============================================================================
# 辅助函数 2 (造轮子): 获取排序后的唯一值集合
# 对应 C++: std::set<double> 或 std::sort 配合 std::unique
# ==============================================================================
matrix_freq_helper_get_unique_sorted <- function(vec) {
  # 这里为了性能使用了 base 库，但在 C++ 中请直接使用 std::set
  out_vec <- base::sort(base::unique(vec))
  return(out_vec)
}

# ==============================================================================
# 辅助函数 3: 提取目标列并构建纯数值型矩阵
# 对应 C++: 将 dataframe 转换为 std::vector<std::vector<double>>
# ==============================================================================
matrix_freq_helper_prep_num_mat <- function(
  stim,
  resp,
  conf = NULL
) {
  n_rows <- base::length(stim)

  # 检查基础向量 stim 和 resp 是否等长
  if (base::length(resp) != n_rows) {
    base::stop("Error: 'stim' and 'resp' must have the same length.")
  }

  # 预分配内存，C++ 的好习惯
  out_mat <- base::matrix(0, nrow = n_rows, ncol = 3)

  # 在 C++ 中，由于强类型限制，这里需要用到 stod 或类型转换
  out_mat[, 1] <- base::as.numeric(stim)
  out_mat[, 2] <- base::as.numeric(resp)

  # 检查信心水平列是否存在
  if (base::is.null(conf)) {
    # 如果为空，将信心列设为常数 1，以退化为经典的信号检测论频数矩阵（如 2x2）
    out_mat[, 3] <- 1
  } else {
    if (base::length(conf) != n_rows) {
      base::stop(
        "Error: 'conf' must have the same length as 'stim' and 'resp'."
      )
    }
    out_mat[, 3] <- base::as.numeric(conf)
  }

  return(out_mat)
}

# ==============================================================================
# 主过程: 计算频数矩阵
# ==============================================================================
matrix_freq <- function(stim, resp, conf = NULL) {
  # 1. 获取清洗后的纯数值矩阵 (列: 1=sig, 2=dec, 3=conf)
  num_mat <- matrix_freq_helper_prep_num_mat(stim, resp, conf)
  n_rows <- base::nrow(num_mat)

  # 2. 提取唯一的信号和信心水平，用于界定最终频数矩阵的维度
  unique_sig <- matrix_freq_helper_get_unique_sorted(num_mat[, 1])
  unique_dec <- matrix_freq_helper_get_unique_sorted(num_mat[, 2])
  unique_conf <- matrix_freq_helper_get_unique_sorted(num_mat[, 3])

  n_sig <- base::length(unique_sig)
  n_dec <- base::length(unique_dec)
  n_conf <- base::length(unique_conf)

  # 3. 预分配频数矩阵
  # 行 = 信号类型数，列 = 决策类型数 * 信心水平数
  n_cols_out <- n_dec * n_conf
  freq_mat <- base::matrix(0, nrow = n_sig, ncol = n_cols_out)

  # 4. 遍历每一行进行计数 (完全的 C++ 风格底层循环)
  for (i in 1:n_rows) {
    cur_sig <- num_mat[i, 1]
    cur_dec <- num_mat[i, 2]
    cur_conf <- num_mat[i, 3]

    # 获取当前值在唯一集合中的相对索引 (1-based)
    row_idx <- matrix_freq_helper_find_index(unique_sig, cur_sig)
    dec_idx <- matrix_freq_helper_find_index(unique_dec, cur_dec)
    conf_idx <- matrix_freq_helper_find_index(unique_conf, cur_conf)

    # 核心逻辑: 将决策与信心二维特征，平铺映射到一维的列索引上
    # 例如: 先排列所有 dec=0 的信心，再排列所有 dec=1 的信心
    col_idx <- (dec_idx - 1) * n_conf + conf_idx

    # 原地频数累加
    freq_mat[row_idx, col_idx] <- freq_mat[row_idx, col_idx] + 1
  }

  # 5. 添加行名和列名，避免纯数字引发数据解析问题
  base::rownames(freq_mat) <- base::paste0("sig_", unique_sig)

  col_names <- base::character(n_cols_out)
  for (d in 1:n_dec) {
    for (c in 1:n_conf) {
      idx <- (d - 1) * n_conf + c
      if (base::is.null(conf)) {
        # 当没有信心列时，仅保留决策前缀（完美还原经典的 2x2 命名法）
        col_names[idx] <- base::paste0("dec_", unique_dec[d])
      } else {
        # 有信心列时，正常拼接 decision 与 confidence
        col_names[idx] <- base::paste0(
          "dec_",
          unique_dec[d],
          "_conf_",
          unique_conf[c]
        )
      }
    }
  }
  base::colnames(freq_mat) <- col_names

  return(freq_mat)
}
