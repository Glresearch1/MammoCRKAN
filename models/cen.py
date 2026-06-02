import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class MAX_model(nn.Module):
    def __init__(self, weights=None, num_classes=3):
        super(MAX_model, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if weights is not None:
            state_dict = torch.load(weights)
            # 删除 `fc` 层的权重，因为它的形状不匹配
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            self.resnet50.load_state_dict(state_dict, strict=False)

        # 修改 fc 层
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

        for param in self.resnet50.parameters():
            param.requires_grad = False  # 冻结 ResNet 的参数

        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward_once(self, view_data):
        # x, boxes, _ = view_data
        x = view_data

        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        # x = torch.cat((x, boxes[:, :4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.normalize(x, p=2, dim=1)

    def forward(self, img):
        view_0 = img['image'][:, 0]  # MLO
        view_1 = img['image'][:, 1]  # CC

        embedd_0 = self.forward_once(view_0)  # 提取 MLO 特征
        embedd_1 = self.forward_once(view_1)  # 提取 CC 特征
        print('embedd_0',embedd_0.shape)
        print('embedd_1',embedd_1.shape)


        # 计算 MLO 和 CC 之间的相似度矩阵
        context = torch.matmul(embedd_0, embedd_1.transpose(-1, -2))
        print('context',context.shape)

        # 取最大相似度值作为最终预测
        preds, _ = torch.max(context, dim=1)  
        print('preds',preds.shape)

        return preds

    



    