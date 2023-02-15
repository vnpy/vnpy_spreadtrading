# 1.1.6版本

1. 改为使用OmsEngine提供的OffsetConverter组件
2. 添加回测中统计绩效时对于爆仓情况的检查

# 1.1.5版本

1. 使用zoneinfo替换pytz库
2. 调整安装脚本setup.cfg，添加Python版本限制

# 1.1.4版本

1. 将模块的图标文件信息，改为完整路径字符串
2. 回测引擎增加暴力穷举和遗传算法的参数优化算法支持
3. SpreadData增加可选参数complie_formula，用于支持回测优化时的公式不编译
