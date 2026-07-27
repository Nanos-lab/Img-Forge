"""TinyCD 模型预测封装。

加载预训练 TinyCD 模型，对图块对进行变化检测推理。
使用模块级单例避免重复加载模型。
"""

from pathlib import Path

import numpy as np
import torch

from app.tools.img_changedet.models import ChangeClassifier

# ImageNet 归一化参数
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD = [0.229, 0.224, 0.225]

# 模块级单例
_predictor = None
_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "pretrained" / "levir_best.pth"


def get_predictor(checkpoint_path: str | None = None) -> "TinyCDPredictor":
    """获取 TinyCD 预测器单例。"""
    global _predictor
    if _predictor is None:
        ckpt = checkpoint_path or str(_DEFAULT_CHECKPOINT)
        _predictor = TinyCDPredictor(ckpt)
    return _predictor


class TinyCDPredictor:
    """TinyCD 模型预测器。"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.model = ChangeClassifier()
        self.model.to(self.device)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"TinyCD model loaded: {param_count:,} params, {device}")

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """将 RGB uint8 图块转为归一化 CHW 张量。"""
        x = img.astype(np.float32) / 255.0
        for c in range(3):
            x[:, :, c] = (x[:, :, c] - _RGB_MEAN[c]) / _RGB_STD[c]
        x = x.transpose(2, 0, 1)
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, old_patch: np.ndarray, new_patch: np.ndarray) -> np.ndarray:
        """对单个图块对进行变化检测。

        Args:
            old_patch: 变化前图块 (H, W, 3) uint8。
            new_patch: 变化后图块 (H, W, 3) uint8。

        Returns:
            概率图 (H, W) float32，0~1，越大变化概率越高。
        """
        ref = self.preprocess(old_patch)
        test = self.preprocess(new_patch)
        output = self.model(ref, test)
        return output.squeeze().cpu().numpy()

    def predict_batch(self, tile_set) -> list[np.ndarray]:
        """对图块集合进行批量推理。

        Args:
            tile_set: TileSet 对象。

        Returns:
            概率图列表，与 tile_set.tiles 顺序一致。
        """
        results = []
        for i, tile in enumerate(tile_set.tiles):
            prob = self.predict(tile.old_patch, tile.new_patch)
            results.append(prob)
            if (i + 1) % 10 == 0:
                print(f"  Inference: {i + 1}/{len(tile_set.tiles)} tiles")
        return results
