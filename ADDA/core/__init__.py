from .adapt import train_tgt
from .pretrain import eval_src, train_src
from .test import eval_tgt
#“.”  代表使用相对路径导入，即从当前项目中寻找需要导入的包或函数

__all__ = (eval_src, train_src, train_tgt, eval_tgt)#all用于导入（）里面包含的变量
