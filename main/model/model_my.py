import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main.model.ResNet3D import generate_model as resnet3D
from main.model.kan import KAN
from main.model.otk.layers import OTKernel



def ini_weights(module_list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Res_3D_Encoder(nn.Module):
    def __init__(self, kargs):
        super().__init__()
        self.model = resnet3D(kargs)
        self.feature_channel = 2048 if kargs.model_depth == 50 else 512

    def forward(self, x, squeeze_to_vector=False, sfusion=False):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        x1, x2, x3, x4 = self.model(x)
        if sfusion:
            return [x1, x2, x3, x4]
        if squeeze_to_vector:
            x = F.adaptive_max_pool3d(x4, 1)
            return torch.flatten(x, start_dim=1)

        x = x4.transpose(1, 2)
        x = F.adaptive_max_pool3d(x, (self.feature_channel, 1, 1))
        return torch.flatten(x, start_dim=2)


class Branch_Classifier(nn.Module):
    def __init__(self, class_num, feature_channel, dropout_rate=0.05):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_channel, class_num),
        )
        ini_weights(self.classifier)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.classifier(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, num_inputs):
        super().__init__()
        concat_channels = in_channels * num_inputs
        self.layer_norm = nn.LayerNorm(concat_channels)
        self.conv1x1x1 = nn.Conv3d(concat_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv1x1x1(x)
        return self.activation(x)


class Co_Plane_Att_2(nn.Module):
    def __init__(self, embed_dim_1, embed_dim_2):
        super().__init__()
        self.emb_dim_1 = embed_dim_1
        self.emb_dim_2 = embed_dim_2

        self.mq1 = nn.Linear(embed_dim_1, embed_dim_1, bias=False)
        self.mk1 = nn.Linear(embed_dim_2, embed_dim_1, bias=False)
        self.mv1 = nn.Linear(embed_dim_2, embed_dim_1, bias=False)

        self.mq2 = nn.Linear(embed_dim_2, embed_dim_2, bias=False)
        self.mk2 = nn.Linear(embed_dim_1, embed_dim_2, bias=False)
        self.mv2 = nn.Linear(embed_dim_1, embed_dim_2, bias=False)

        self.norm = nn.LayerNorm(embed_dim_1)
        ini_weights(self.modules())

    def forward(self, feature_1, feature_2):
        q1 = self.mq1(feature_1)
        k1 = self.mk1(feature_2).permute(0, 2, 1)
        v1 = self.mv1(feature_2)
        att1 = torch.softmax(torch.matmul(q1, k1) / np.sqrt(self.emb_dim_1), dim=-1)
        out1 = torch.matmul(att1, v1)

        q2 = self.mq2(feature_2)
        k2 = self.mk2(feature_1).permute(0, 2, 1)
        v2 = self.mv2(feature_1)
        att2 = torch.softmax(torch.matmul(q2, k2) / np.sqrt(self.emb_dim_2), dim=-1)
        out2 = torch.matmul(att2, v2)

        fused = self.norm(out1 + feature_1 + out2 + feature_2)
        fused = fused.transpose(1, 2)
        fused = F.adaptive_max_pool1d(fused, 1)
        return torch.flatten(fused, start_dim=1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class STMRI(nn.Module):
    def __init__(self, kargs):
        super().__init__()
        self.kargs = kargs
        self.encoder = Res_3D_Encoder(kargs)
        self.feature = int(self.encoder.feature_channel)

        self.classifier = Branch_Classifier(kargs.class_num, self.feature)
        self.classifier_kan = KAN([
            self.feature,
            self.feature // 2,
            self.feature // 4,
            self.feature // 8,
            kargs.class_num,
        ])

        self.lstm_model = LSTMClassifier(
            input_size=self.feature,
            hidden_size=self.feature,
            num_classes=kargs.class_num,
        )

        if kargs.model_depth == 50:
            self.dims = [256, 512, 1024, 2048]
        else:
            self.dims = [64, 128, 256, 512]

        num_inputs = 2 if kargs.oadce == 2 else 4
        self.stages_fusion = nn.ModuleList([
            FusionBlock(in_channels=d, num_inputs=num_inputs) for d in self.dims
        ])

        self.latlayer3 = nn.Conv3d(self.dims[-1], self.dims[-2], kernel_size=1)
        self.latlayer2 = nn.Conv3d(self.dims[-2], self.dims[-3], kernel_size=1)
        self.latlayer1 = nn.Conv3d(self.dims[-3], self.dims[-4], kernel_size=1)
        self.latlayer0 = nn.Conv3d(self.dims[-4], self.dims[-4], kernel_size=1)

        self.otk_norm1 = nn.BatchNorm3d(self.dims[1])
        self.otk_norm2 = nn.BatchNorm3d(self.dims[0])
        self.otk_pool1 = nn.Conv3d(self.dims[1], self.dims[1], kernel_size=3, padding=1)
        self.otk_pool2 = nn.Conv3d(self.dims[0], self.dims[0], kernel_size=3, padding=1)

        self.norm_fpn0 = nn.LayerNorm(self.dims[2], eps=1e-6)
        self.norm_fpn1 = nn.LayerNorm(self.dims[1], eps=1e-6)
        self.norm_fpn2 = nn.LayerNorm(self.dims[0], eps=1e-6)

        self.otk1 = nn.Sequential(nn.ReLU(), OTKernel(in_dim=self.dims[1], out_size=self.dims[1], heads=1))
        self.otk2 = nn.Sequential(nn.ReLU(), OTKernel(in_dim=self.dims[0], out_size=self.dims[0], heads=1))
        self.otk3 = nn.Sequential(
            nn.ReLU(),
            OTKernel(in_dim=self.dims[0], out_size=self.dims[1], heads=1),
            OTKernel(in_dim=self.dims[0], out_size=self.dims[0], heads=1),
        )

        self.cross_attention = Co_Plane_Att_2(self.feature, self.dims[3])

    def _upsample_add(self, x, y):
        _, _, d, h, w = y.size()
        return F.interpolate(x, size=[d, h, w], mode='trilinear', align_corners=False) + y

    def _spatial_fusion(self, feat_bu):
        b = feat_bu[-1].shape[0]
        fpn0 = self.latlayer3(feat_bu[-1])
        fpn1 = self._upsample_add(self.latlayer2(fpn0), self.latlayer2(feat_bu[-2]))
        fpn2 = self._upsample_add(self.latlayer1(fpn1), self.latlayer1(feat_bu[-3]))
        fpn3 = self._upsample_add(self.latlayer0(fpn2), self.latlayer0(feat_bu[-4]))

        fpn1 = self.otk_norm1(self.otk_pool1(fpn1))
        fpn2 = self.otk_norm2(self.otk_pool2(fpn2))
        fpn3 = self.otk_norm2(self.otk_pool2(fpn3))

        fpn0 = fpn0.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, self.dims[2])
        fpn1 = fpn1.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, self.dims[1])
        fpn2 = fpn2.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, self.dims[0])
        fpn3 = fpn3.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, self.dims[0])

        fpn1 = self.otk1(fpn1)
        fpn2 = self.otk2(fpn2)
        fpn3 = self.otk3(fpn3)

        fpn0 = self.norm_fpn0(fpn0.mean(-2))
        fpn1 = self.norm_fpn1(fpn1.mean(-2))
        fpn2 = self.norm_fpn2(fpn2.mean(-2))
        fpn3 = self.norm_fpn2(fpn3.mean(-2))
        return torch.cat([fpn0, fpn1, fpn2, fpn3], 1)

    def _temporal_fusion(self, time_sequence):
        pooled = []
        for x in time_sequence:
            x = F.adaptive_max_pool3d(x, 1)
            pooled.append(torch.flatten(x, start_dim=1))
        time_sequence = torch.stack(pooled, dim=1)
        return self.lstm_model(time_sequence)

    def forward(self, input_dict):
        phases = input_dict.get('ph', [])
        required = 2 if self.kargs.oadce == 2 else 4
        if len(phases) < required:
            raise ValueError(f'STMRI requires at least {required} phase volumes, got {len(phases)}')

        feats = [self.encoder(x, sfusion=True) for x in phases[:required]]

        pred_s, pred_t = None, None

        if self.kargs.branch in ('ST', 'S'):
            s_sequence = []
            for i in range(4):
                stage_inputs = [f[i] for f in feats]
                s_sequence.append(self.stages_fusion[i](stage_inputs))
            s_out = self._spatial_fusion(s_sequence)
            pred_s = self.classifier_kan(s_out) if self.kargs.classtool == 'kan' else self.classifier(s_out)
            if self.kargs.branch == 'S':
                return [pred_s] * 4

        if self.kargs.branch in ('ST', 'T'):
            time_sequence = [f[3] for f in feats]
            t_out = self._temporal_fusion(time_sequence)
            pred_t = self.classifier_kan(t_out) if self.kargs.classtool == 'kan' else self.classifier(t_out)
            if self.kargs.branch == 'T':
                return [pred_t] * 4

        st_out = self.cross_attention(t_out.unsqueeze(1), s_out.unsqueeze(1))
        pred_st = self.classifier_kan(st_out) if self.kargs.classtool == 'kan' else self.classifier(st_out)

        if self.kargs.mloss:
            return [pred_st, pred_s, pred_t]
        return [pred_st] * 4

    def criterion(self, pred, label):
        if self.kargs.class_num > 1:
            lossfunc = nn.CrossEntropyLoss()
            label = label.long().view(-1)
        else:
            lossfunc = nn.BCEWithLogitsLoss()
            label = label.float().view(-1, 1)

        if len(pred) == 3:
            pred_st, pred_s, pred_t = pred
            loss = 0.4 * (
                lossfunc(pred_s, label) +
                lossfunc(pred_t, label) +
                lossfunc(pred_st, label)
            )
            return loss, float(loss.item())

        final_pred = pred[0]
        loss = lossfunc(final_pred, label)
        return loss, float(loss.item())
