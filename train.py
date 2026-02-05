import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 必须位于其他库导入前
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import mobile_vit_small
from utils import read_train_val_data, train_one_epoch, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_train_val_data(args.data_path)



    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 创建修改后的模型
    model = mobile_vit_small(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]

        # 根据修改后的模型结构调整权重加载逻辑
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        print(f"成功加载 {len(pretrained_dict)} 个参数")
        skipped_params = set(model_dict.keys()) - set(pretrained_dict.keys())
        print(f"跳过 {len(skipped_params)} 个形状不匹配的参数")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-2)

    # SGD + 动量 + 权重衰减
    # optimizer = optim.SGD(
    #     model.parameters(), lr=0.0002, momentum=0.9, weight_decay=5e-4
    # )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7)
    # 使用 CosineAnnealingWarmRestarts 学习率调节器
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            num_classes=args.num_classes
        )

        # validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            num_classes=args.num_classes
        )

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 记录到TensorBoard
        tags = ["train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "learning_rate"]

        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_precision, epoch)
        tb_writer.add_scalar(tags[3], train_recall, epoch)
        tb_writer.add_scalar(tags[4], train_f1, epoch)
        tb_writer.add_scalar(tags[5], val_loss, epoch)
        tb_writer.add_scalar(tags[6], val_acc, epoch)
        tb_writer.add_scalar(tags[7], val_precision, epoch)
        tb_writer.add_scalar(tags[8], val_recall, epoch)
        tb_writer.add_scalar(tags[9], val_f1, epoch)
        tb_writer.add_scalar(tags[10], optimizer.param_groups[0]["lr"], epoch)

        # 打印本轮结果
        print(f"Epoch {epoch}: "
              f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%, "
              f"Train Precision: {train_precision:.3f}%, Train Recall: {train_recall:.3f}%, Train F1: {train_f1:.3f}%, "
              f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}%, "
              f"Val Precision: {val_precision:.3f}%, Val Recall: {val_recall:.3f}%, Val F1: {val_f1:.3f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

        # 保存最新模型
        torch.save(model.state_dict(), "./weights/latest_model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("--data_path", default="./obc622", help="dataset path")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--weights", default="./weights/mobilevit_s.pt", help="pretrained weights path")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--num_classes", default=306, type=int, help="number of classes")

    args = parser.parse_args()

    main(args)