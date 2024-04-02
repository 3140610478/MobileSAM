import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from config import n_1, n_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegAcc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return ((predicted>0.5) == actual).to(torch.float32).mean()
    

class MIoU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor, precision: float=0.0001)->torch.Tensor:
        i1 = ((predicted>0.5) * actual)
        u1 = ((predicted>0.5) + actual) - i1
        i1, u1 = i1.sum(dim=(2,3))+precision, u1.sum(dim=(2,3))+precision
        i0 = ((predicted<=0.5) * (1-actual))
        u0 = ((predicted<=0.5) + (1-actual)) - i0
        i0, u0 = i0.sum(dim=(2,3))+precision, u0.sum(dim=(2,3))+precision
        return (i1/u1 + i0/u0).to(torch.float32).mean()/2


class BoundaryLoss(torch.nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5,nc=2):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.num_class=nc

    def crop(self, w, h, target):
        nt, ht, wt = target.size()
        offset_w, offset_h = (wt - w) // 2, (ht - h) // 2
        if offset_w > 0 and offset_h > 0:
            target = target[:, offset_h:-offset_h, offset_w:-offset_w]

        return target

    def to_one_hot(self, target, size):
        # n, c, h, w = size
        # if c==1:
        #     c=self.num_class
        # ymask = torch.FloatTensor(size).zero_()
        # new_target = torch.LongTensor(n, 1, h, w)
        # # print()
        # # if target.is_cuda:
        # #     ymask = ymask.cuda(target.get_device())
        #     # new_target = new_target.cuda(target.get_device())

        # new_target[:, 0, :, :] = torch.clamp(target.detach(), 0, c - 1)#target裁切到[0,1]范围
        # ymask.scatter_(1, new_target, 1)#dim=1,按照行排列
        # y_list=ymask.tolist()
        return 1-target
        # return torch.autograd.Variable(ymask).to(device)

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        gt = torch.squeeze(gt, dim=1)

        n, c, h, w = pred.shape
        log_p = F.log_softmax(pred, dim=1)

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        gt = self.crop(w, h, gt)
        one_hot_gt = self.to_one_hot(gt, log_p.size()).to(torch.float32)
        # print((1-one_hot_gt).unique())
        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt#经过验证，的确为boundary map

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class Conbined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BF = BoundaryLoss()
        self.bce = BCELoss()

    def forward(self, y_pred, y):
        return n_1*self.bce(y_pred, y.to(torch.float32))+n_2*self.BF(y_pred, y)


loss_fun = Conbined()
acc_fun = SegAcc()
miou_fun = MIoU()
