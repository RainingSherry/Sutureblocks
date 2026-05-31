


# 测试自适应尺度选择模块
if __name__ == '__main__':
    block = AdaptiveScaleMLA(in_channels=64, out_channels=64)  # 通过选择不同的卷积尺度
    input1 = torch.rand(3, 64, 32, 32)  # 输入尺寸为 32x32
    output = block(input1)
    print(input1.size())
    print(output.size())
    input2 = torch.rand(3, 64, 256, 256)  # 输入尺寸为 256x256
    output = block(input2)
    print(input2.size())
    print(output.size())
