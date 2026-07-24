"""ImgForge 应用入口 —— 模块化遥感影像处理 API。"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import ImageProcessingError
from app.tools.img_enhance import router as enhance_router
from app.tools.obb_detect import router as obb_detect_router
from app.tools.img_mosaic import router as mosaic_router
from app.tools.img_ortho import router as ortho_router

app = FastAPI(
    title="ImgForge",
    version="0.4.0",
    description="模块化遥感影像处理 API —— 影像增强、目标检测、影像拼接、正射校正等工具集",
)

# ---- 注册工具路由 ----
app.include_router(enhance_router)
app.include_router(obb_detect_router)
app.include_router(mosaic_router)
app.include_router(ortho_router)


# ---- 全局异常处理 ----
@app.exception_handler(ImageProcessingError)
async def image_processing_exception_handler(request: Request, exc: ImageProcessingError):
    """统一处理图片处理相关的业务异常。"""
    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "message": exc.message,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """兜底异常处理。"""
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": f"服务器内部错误: {exc}",
        },
    )


@app.get("/", tags=["健康检查"])
async def root():
    """服务健康检查。"""
    return {
        "service": "ImgForge",
        "version": "0.4.0",
        "status": "running",
    }
