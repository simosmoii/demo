import uvicorn
import webbrowser
import threading
import time

def open_browser():
    """
    延时1.5秒后自动打开浏览器，
    确保在这1.5秒内 FastAPI 后端服务已经成功启动完毕。
    """
    time.sleep(1.5)
    print("\n正在打开浏览器访问系统主页...\n")
    webbrowser.open("http://127.0.0.1:8000/")

if __name__ == "__main__":
    # 1. 开启一个后台子线程去执行打开浏览器的任务
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 2. 启动后端的 FastAPI 服务
    print("🚀 正在启动鸟类识别系统后端服务...")
    # 注意：这里使用模块字符串 "main:app" 是 uvicorn 的标准推荐写法
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)