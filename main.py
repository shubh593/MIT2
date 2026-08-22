import sys
import uvicorn

def main():
    print("==================================================================")
    print(" [AEROSAFE-LUNAR] AI Planetary Landing Risk Assessment (SW08)")
    print(" Dual AI Engine: ONNX Runtime (best.onnx) & PyTorch YOLO (best.pt)")
    print(" Dashboard URL: http://localhost:8000")
    print(" API Documentation: http://localhost:8000/docs")
    print("==================================================================")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
