import argparse
import subprocess
import os
import sys

def run_data_pipeline():
    print("--- Chạy Data Pipeline ---")
    script_path = os.path.join("src", "data_pipeline", "data_loader.py")
    if os.path.exists(script_path):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Chạy trong thư mục src/data_pipeline để đảm bảo đường dẫn dữ liệu tương đối (../../data/...) hoạt động đúng
        cwd = os.path.join(os.getcwd(), "src", "data_pipeline")
        try:
            subprocess.run([sys.executable, "data_loader.py"], check=True, env=env, cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f"!!! Lỗi khi chạy Data Pipeline: {e} !!!")
    else:
        print(f"!!! Không tìm thấy file {script_path} !!!")
    print("\n")

def run_ml_pipeline():
    print("--- Chạy ML Pipeline ---")
    script_path = os.path.join("src", "ml_pipeline", "train.py")
    if os.path.exists(script_path):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Chạy trong thư mục src/ml_pipeline để models/ được lưu đúng chỗ
        cwd = os.path.join(os.getcwd(), "src", "ml_pipeline")
        try:
            subprocess.run([sys.executable, "train.py"], check=True, env=env, cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f"!!! Lỗi khi chạy ML Pipeline: {e} !!!")
    else:
        print(f"!!! Không tìm thấy file {script_path} !!!")
    print("\n")

def build_vector_db():
    print("--- Khởi tạo VB (ChromaDB) ---")
    vector_py_path = os.path.join("src", "rag_pipeline", "vector_store.py")
    if os.path.exists(vector_py_path):
        # Thiết lập PYTHONPATH để nhận diện thư mục src
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run([sys.executable, vector_py_path], check=True, env=env)
    else:
        print(f"!!! Không tìm thấy file {vector_py_path} !!!")

def run_streamlit():
    print("--- Khởi chạy Streamlit Dashboard ---")
    app_py_path = os.path.join("app", "main.py")
    if os.path.exists(app_py_path):
        # Thiết lập PYTHONPATH để nhận diện thư mục src
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(["streamlit", "run", app_py_path], env=env)
    else:
        print(f"!!! Không tìm thấy file {app_py_path} !!!")

def setup_ollama():
    print("--- Đang tải Model Llama3 cho Ollama ---")
    print("Lưu ý: Bạn cần cài đặt ứng dụng Ollama trước khi chạy lệnh này.")
    try:
        # Chạy lệnh pull để tải model llama3
        subprocess.run(["ollama", "pull", "llama3"], check=True)
        print("Đã tải xong model llama3!")
    except FileNotFoundError:
        print("!!! Error: Không tìm thấy lệnh 'ollama'. Vui lòng cài đặt Ollama tại https://ollama.com/ !!!")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error khi tải model: {e} !!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCCRA Project Runner")
    parser.add_argument("--all", action="store_true", help="Chạy mọi thứ")
    parser.add_argument("--data", action="store_true", help="Kiểm tra Data pipeline")
    parser.add_argument("--ml", action="store_true", help="Kiểm tra ML pipeline")
    parser.add_argument("--rag", action="store_true", help="Chạy mã xây dựng Vector DB")
    parser.add_argument("--app", action="store_true", help="Chạy giao diện ứng dụng Streamlit")
    parser.add_argument("--setup-local", action="store_true", help="Tải model Llama3 cho Ollama (chạy dưới máy)")
    
    args = parser.parse_args()
    
    # Nếu không truyền tham số nào, mặc định chạy Web App
    if not (args.all or args.data or args.ml or args.rag or args.app or args.setup_local):
        args.app = True

    # --- KIỂM TRA FILE TRƯỚC KHI CHẠY ---
    db_path = os.path.join("data", "database", "sccra_db.sqlite")
    model_path = os.path.join("src", "ml_pipeline", "models", "product_classifier.pkl")
    
    # Nếu chạy app mà thiếu DB hoặc Model, tự động gợi ý chạy --all
    if args.app and not args.all:
        if not os.path.exists(db_path):
            print(f"!!! Cảnh báo: Không tìm thấy Database tại {db_path} !!!")
            choice = input("Bạn có muốn chạy Data Pipeline để tạo mới không? (y/n): ")
            if choice.lower() == 'y':
                args.data = True
        
        if not os.path.exists(model_path):
            print(f"!!! Cảnh báo: Không tìm thấy Model tại {model_path} !!!")
            choice = input("Bạn có muốn chạy ML Pipeline để huấn luyện model không? (y/n): ")
            if choice.lower() == 'y':
                args.ml = True

    if args.setup_local:
        setup_ollama()

    if args.all or args.data:
        run_data_pipeline()
        
    if args.all or args.ml:
        run_ml_pipeline()
        
    if args.all or args.rag:
        try:
            build_vector_db()
        except subprocess.CalledProcessError:
            print("!!! Lỗi khi khởi tạo Vector DB. Kiểm tra lại môi trường. !!!")
        
    if args.all or args.app:
        run_streamlit()
