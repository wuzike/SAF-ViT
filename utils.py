import os
import json
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def read_train_val_data(root: str):
    """读取训练集和验证集数据"""
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    # 验证路径存在
    assert os.path.exists(train_root), f"训练集路径不存在: {train_root}"
    assert os.path.exists(val_root), f"验证集路径不存在: {val_root}"

    def get_images_and_labels(data_root):
        """获取指定目录下的所有图片路径和标签"""
        # 获取所有类别文件夹
        classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        classes.sort()  # 排序以确保类别顺序一致

        # 生成类别到索引的映射
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # 保存映射关系到JSON文件
        with open('class_indices1.json', 'w') as f:
            json.dump({i: cls for cls, i in class_to_idx.items()}, f, indent=4)

        images_path = []  # 存储图片路径
        images_label = []  # 存储图片标签

        # 支持的图片文件扩展名
        supported_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG','.bmp']

        # 遍历每个类别文件夹
        for cls in classes:
            cls_dir = os.path.join(data_root, cls)
            # 获取该类别下的所有图片文件
            for img_name in os.listdir(cls_dir):
                if any(img_name.endswith(ext) for ext in supported_extensions):
                    img_path = os.path.join(cls_dir, img_name)
                    images_path.append(img_path)
                    images_label.append(class_to_idx[cls])

        print(f"在 {data_root} 中找到 {len(images_path)} 张图片")
        return images_path, images_label

    # 读取训练集和验证集
    train_images_path, train_images_label = get_images_and_labels(train_root)
    val_images_path, val_images_label = get_images_and_labels(val_root)

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_test_data(root: str):
    """读取测试集数据"""
    test_root = root  # 测试集根目录直接包含类别文件夹

    # 验证路径存在
    assert os.path.exists(test_root), f"测试集路径不存在: {test_root}"

    def get_images_and_labels(data_root):
        """获取指定目录下的所有图片路径和标签"""
        # 获取所有类别文件夹
        classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        classes.sort()  # 排序以确保类别顺序一致

        # 生成类别到索引的映射
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        images_path = []  # 存储图片路径
        images_label = []  # 存储图片标签

        # 支持的图片文件扩展名
        supported_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG','.bmp']

        # 遍历每个类别文件夹
        for cls in classes:
            cls_dir = os.path.join(data_root, cls)
            # 获取该类别下的所有图片文件
            for img_name in os.listdir(cls_dir):
                if any(img_name.endswith(ext) for ext in supported_extensions):
                    img_path = os.path.join(cls_dir, img_name)
                    images_path.append(img_path)
                    images_label.append(class_to_idx[cls])

        print(f"在测试集中找到 {len(images_path)} 张图片")
        return images_path, images_label

    # 读取测试集
    test_images_path, test_images_label = get_images_and_labels(test_root)

    return test_images_path, test_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = 'class_indices1.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def accuracy(output, target):
    """计算top-1准确率"""
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=False)
        return correct_k


def calculate_metrics(preds, targets, num_classes):
    """计算Precision、Recall和F1-score"""
    # 转换为numpy数组
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # 计算宏平均精确率、召回率和F1分数
    precision = precision_score(targets_np, preds_np, average='macro', labels=range(num_classes)) * 100
    recall = recall_score(targets_np, preds_np, average='macro', labels=range(num_classes)) * 100
    f1 = f1_score(targets_np, preds_np, average='macro', labels=range(num_classes)) * 100

    return precision, recall, f1


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = 0  # 累计预测正确的样本数
    optimizer.zero_grad()

    all_preds = []
    all_targets = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 保存所有预测和目标值用于计算指标
        all_targets.append(labels)

        pred = model(images.to(device))
        _, preds = pred.topk(1, 1, True, True)
        all_preds.append(preds.cpu())

        correct = accuracy(pred, labels.to(device))
        accu_num += correct.item()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        # 计算当前准确率
        current_acc = (accu_num / sample_num) * 100

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}%".format(
            epoch, accu_loss.item() / (step + 1), current_acc)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 计算Precision, Recall, F1-score
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0)
    precision, recall, f1 = calculate_metrics(all_preds, all_targets, num_classes)

    return (accu_loss.item() / (step + 1),
            (accu_num / sample_num) * 100,
            precision, recall, f1)

# def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes):
#     model.train()
#     loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = 0  # 累计预测正确的样本数
#     optimizer.zero_grad()
#
#     all_preds = []
#     all_targets = []
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#
#         # 保存所有目标值用于计算指标
#         all_targets.append(labels)
#
#         # 1. 获取模型输出（GoogLeNet返回元组：(主输出, 辅助输出1, 辅助输出2)）
#         outputs = model(images.to(device))
#         # 提取主分类器输出用于预测
#         pred_main = outputs[0]
#
#         # 2. 计算预测结果（使用主输出）
#         _, preds = pred_main.topk(1, 1, True, True)
#         all_preds.append(preds.cpu())
#
#         # 3. 计算准确率（使用主输出）
#         correct = accuracy(pred_main, labels.to(device))
#         accu_num += correct.item()
#
#         # 4. 计算损失（主输出损失 + 辅助输出损失，增强训练稳定性）
#         loss = loss_function(pred_main, labels.to(device))
#         # 如果有辅助分类器输出，添加辅助损失（权重0.3是常见设置）
#         if len(outputs) > 1:
#             loss += 0.3 * loss_function(outputs[1], labels.to(device))
#             loss += 0.3 * loss_function(outputs[2], labels.to(device))
#
#         loss.backward()
#         accu_loss += loss.detach()
#
#         # 计算当前准确率
#         current_acc = (accu_num / sample_num) * 100
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}%".format(
#             epoch, accu_loss.item() / (step + 1), current_acc)
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#     # 计算Precision, Recall, F1-score
#     all_preds = torch.cat(all_preds, dim=0).squeeze()
#     all_targets = torch.cat(all_targets, dim=0)
#     precision, recall, f1 = calculate_metrics(all_preds, all_targets, num_classes)
#
#     return (accu_loss.item() / (step + 1),
#             (accu_num / sample_num) * 100,
#             precision, recall, f1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = 0  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    all_preds = []
    all_targets = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 保存所有预测和目标值用于计算指标
        all_targets.append(labels)

        pred = model(images.to(device))
        _, preds = pred.topk(1, 1, True, True)
        all_preds.append(preds.cpu())

        correct = accuracy(pred, labels.to(device))
        accu_num += correct.item()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 计算当前准确率
        current_acc = (accu_num / sample_num) * 100

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}%".format(
            epoch, accu_loss.item() / (step + 1), current_acc)

    # 计算Precision, Recall, F1-score
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0)
    precision, recall, f1 = calculate_metrics(all_preds, all_targets, num_classes)

    return (accu_loss.item() / (step + 1),
            (accu_num / sample_num) * 100,
            precision, recall, f1)