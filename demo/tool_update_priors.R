.update_priors <- function(x, priors) {
  
################################# [prepare] ####################################
  
  free_params <- list()
  posteriors <- list()
  
  # 把自由参数找出来
  for (i in 1:length(x)) {
    free_params[[i]] <- x[[i]]@input@params@free
  }
  # 把自由参数变成表格
  free_params <- .rbind_fill(free_params)
  
  func_names <- list()
  for (i in 1:length(priors)) {
    func_names[[i]] <- .get_dfunc(func = priors[[i]])
  }
  
################################## [update] ####################################
   
  for (i in 1:length(priors)) {
    # 更新后验分布
    posteriors[[i]] <- .create_dfunc(
      df_params = free_params[[i]], 
      func_name = func_names[[i]]
    )
  }
  
  names(posteriors) <- names(priors)

  return(posteriors)
}
