"""自定义异常类。"""


class ImageProcessingError(Exception):
    """图片处理异常基类。"""

    def __init__(self, message: str = "影像处理过程中发生错误"):
        self.message = message
        super().__init__(message)


class InvalidImageError(ImageProcessingError):
    """无效或损坏的影像文件。"""

    def __init__(self, detail: str = "无效的影像文件"):
        self.detail = detail
        super().__init__(f"影像文件无效: {detail}")


class UnsupportedFormatError(InvalidImageError):
    """不支持的影像格式。"""

    def __init__(self, filename: str, allowed: set):
        self.filename = filename
        self.allowed = allowed
        super().__init__(f"不支持的文件格式 '{filename}'，允许的格式: {allowed}")


class FileTooLargeError(ImageProcessingError):
    """上传文件过大。"""

    def __init__(self, actual_size: int, max_size: int):
        self.actual_size = actual_size
        self.max_size = max_size
        super().__init__(
            f"文件大小 ({actual_size / 1024 / 1024:.1f} MB) "
            f"超过限制 ({max_size / 1024 / 1024:.1f} MB)"
        )


class EnhancementError(ImageProcessingError):
    """影像增强处理失败。"""

    def __init__(self, detail: str = "增强处理失败"):
        self.detail = detail
        super().__init__(f"增强处理失败: {detail}")


class DetectionError(ImageProcessingError):
    """目标检测处理失败。"""

    def __init__(self, detail: str = "目标检测失败"):
        self.detail = detail
        super().__init__(f"目标检测失败: {detail}")


class MosaicError(ImageProcessingError):
    """影像拼接处理失败。"""

    def __init__(self, detail: str = "影像拼接失败"):
        self.detail = detail
        super().__init__(f"影像拼接失败: {detail}")


class OrthoError(ImageProcessingError):
    """正射校正处理失败。"""

    def __init__(self, detail: str = "正射校正失败"):
        self.detail = detail
        super().__init__(f"正射校正失败: {detail}")


