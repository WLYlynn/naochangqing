# main.py - 用于 Buildozer 打包
from app import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)