import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchattacks import PGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Đảm bảo đường dẫn này đúng với cấu trúc project của bạn
from CIFAR10.Train.arch.RN_FC import VAE as FLSS 

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm tải mô hình
# @st.cache_resource sử dụng để cache model, tránh tải lại mỗi khi tương tác UI
def load_models():
    # Tải ResNet-18 (giả định đã huấn luyện với PGD-AT)
    resnet = torchvision.models.resnet18(pretrained=False)
    # Modify the first conv layer to match the saved weights for CIFAR10 (32x32 images)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()  # Remove maxpool since we're using stride=1 in first conv
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # CIFAR-10: 10 lớp
    
    try:
        # Load state dict và xử lý các khóa không khớp
        # map_location đảm bảo model được load lên đúng device
        state_dict = torch.load("saved_checkpoints/resnet18_pgd_at_epoch_20.pth", map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('layer'):
                # Thay thế 'shortcut' bằng 'downsample' trong các khóa nếu có
                k = k.replace('shortcut', 'downsample')
            new_state_dict[k] = v
        resnet.load_state_dict(new_state_dict)
        st.success("ResNet checkpoint loaded successfully!")
    except FileNotFoundError:
        st.warning("ResNet checkpoint not found. Using randomly initialized ResNet-18.")
        # Nếu không tìm thấy checkpoint, sử dụng ResNet-18 mặc định (chưa huấn luyện)
        # Hoặc bạn có thể tải một phiên bản pretrained khác nếu muốn
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    except Exception as e:
        st.error(f"Error loading ResNet: {e}. Using randomly initialized ResNet-18.")
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    
    resnet = resnet.to(device) # Chuyển model lên đúng device
    resnet.eval() # Đặt model ở chế độ evaluation
    
    # Tải FLSS
    flss = FLSS(latent_dim=512, num_classes=10) # Khởi tạo model FLSS
    try:
        # Load state dict và bỏ tiền tố 'module.' nếu model được huấn luyện với DataParallel
        state_dict = torch.load("saved_checkpoints/FLSS_cifar10.pt", map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        flss.load_state_dict(new_state_dict)
        st.success("FLSS checkpoint loaded successfully!")
    except FileNotFoundError:
        st.warning("FLSS checkpoint not found. Using randomly initialized FLSS.")
    except Exception as e:
        st.error(f"Error loading FLSS: {e}")
    
    flss = flss.to(device) # Chuyển model lên đúng device
    flss.eval() # Đặt model ở chế độ evaluation
    
    return resnet, flss

# Hàm tạo ảnh đối kháng bằng PGD
def generate_adversarial_image(model, image, label, epsilon, alpha, steps):
    # Đảm bảo ảnh và nhãn đều ở trên cùng một device với model
    image = image.to(device)
    label = label.to(device)
    
    attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    adv_image = attack(image, label)
    return adv_image

# Hàm trực quan hóa không gian tiềm ẩn
def visualize_latent_space(mu, logvar, n_samples=100):
    latent_samples = []
    with torch.no_grad(): # Không tính gradient trong phần này
        for _ in range(n_samples):
            # Lấy mẫu từ phân phối tiềm ẩn
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            latent_samples.append(z.squeeze().detach().cpu().numpy()) # Chuyển về CPU và numpy
    
    latent_samples = np.array(latent_samples) # Chuyển danh sách thành mảng numpy
    
    # Áp dụng PCA để giảm chiều về 2D cho việc hiển thị
    pca = PCA(n_components=2)
    samples_2d = pca.fit_transform(latent_samples)
    mu_2d = pca.transform(mu.detach().cpu().numpy())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.5, label='Sampled Features (z)')
    ax.scatter(mu_2d[:, 0], mu_2d[:, 1], color='red', s=100, marker='x', label='Mean Feature (μ)')
    ax.set_title("Latent Space Feature Cloud")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# Hàm trực quan hóa phân phối dự đoán
def visualize_predictions(predictions, class_names):
    counts = np.bincount(predictions, minlength=len(class_names)) # Đếm số lần xuất hiện của mỗi lớp
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, counts, alpha=0.7)
    ax.set_title("Distribution of 100 Sampled Predictions")
    ax.set_ylabel("Frequency")
    
    # Hiển thị số lượng trên mỗi cột
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   str(count), ha='center', va='bottom')
    
    plt.xticks(rotation=45) # Xoay nhãn trục X để dễ đọc
    plt.tight_layout() # Tự động điều chỉnh layout
    return fig

# Hàm chạy demo FLSS
def run_flss_demo(flss, image, n_samples, threshold):
    image = image.to(device) # Đảm bảo ảnh ở đúng device
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    predictions_list = []
    mu_list = []
    logvar_list = []
    
    with torch.no_grad(): # Không tính gradient
        # Lấy mẫu nhiều dự đoán
        for i in range(n_samples):
            try:
                # Thực hiện forward pass của FLSS
                output = flss(image)
                if isinstance(output, tuple) and len(output) >= 4:
                    predictions, mu, logvar, eps = output
                    pred_class = torch.argmax(predictions, dim=1).cpu().numpy()[0]
                    predictions_list.append(pred_class)
                    
                    if i == 0: # Chỉ lưu mu và logvar một lần cho việc trực quan hóa
                        mu_list = mu
                        logvar_list = logvar
                else:
                    # Trường hợp fallback: nếu FLSS không trả về đủ 4 giá trị
                    if isinstance(output, tuple):
                        predictions = output[0]
                    else:
                        predictions = output
                    pred_class = torch.argmax(predictions, dim=1).cpu().numpy()[0]
                    predictions_list.append(pred_class)
            except Exception as e:
                st.warning(f"FLSS prediction error on sample {i}: {e}. Using random prediction as fallback.")
                # Fallback: dự đoán ngẫu nhiên nếu có lỗi
                predictions_list.append(np.random.randint(0, 10))
    
    predictions_array = np.array(predictions_list) # Chuyển danh sách dự đoán thành mảng numpy
    
    # Trực quan hóa
    latent_fig = None
    try:
        if len(mu_list) > 0 and len(logvar_list) > 0:
            latent_fig = visualize_latent_space(mu_list, logvar_list, n_samples)
        else:
            st.info("μ and logvar not available for latent space visualization.")
    except Exception as e:
        st.warning(f"Latent visualization error: {e}")
        latent_fig = None
    
    pred_fig = visualize_predictions(predictions_array, class_names)
    
    # Đưa ra quyết định
    majority_class_idx = np.argmax(np.bincount(predictions_array)) # Lớp có số phiếu cao nhất
    majority_class_name = class_names[majority_class_idx]
    frequency = np.bincount(predictions_array)[majority_class_idx] / len(predictions_array)
    
    decision = "ACCEPT" if frequency >= threshold else "REJECT" # Quyết định dựa trên tần suất và ngưỡng
    result = f"Lớp đa số phiếu: {majority_class_name}\nTần suất (Độ tin cậy): {frequency:.2f}\nNgưỡng loại bỏ: {threshold}\nQuyết định: {decision}"
    
    return latent_fig, pred_fig, result

# Streamlit App
st.set_page_config(layout="wide") # Đặt layout rộng hơn
st.title("FLSS vs ResNet-18: Robustness Demo on CIFAR-10")
st.write(f"Running on: {device}")

# --- Sidebar cho các tham số ---
st.sidebar.header("Tham số tấn công đối kháng (PGD)")
epsilon = st.sidebar.slider("Epsilon (cường độ nhiễu)", 0.0, 0.5, 25/255, step=1/255, 
                            help="Phạm vi tối đa của nhiễu đối kháng (e.g., 8/255).")
steps = st.sidebar.slider("Số bước tấn công", 1, 20, 10, 
                          help="Số lần lặp để tạo ảnh đối kháng. Số bước cao hơn tạo ra ảnh 'mạnh' hơn nhưng tốn thời gian hơn.")

st.sidebar.header("Tham số FLSS")
threshold = st.sidebar.slider("Ngưỡng từ chối (Confidence)", 0.0, 1.0, 0.4, step=0.05,
                              help="Nếu tần suất của lớp đa số dưới ngưỡng này, dự đoán sẽ bị từ chối.")
# n_samples = st.sidebar.slider("Số lượng mẫu (FLSS)", 10, 200, 100, step=10,
#                               help="Số lượng lần lấy mẫu từ không gian tiềm ẩn để đưa ra dự đoán cuối cùng. Số mẫu cao hơn cho kết quả ổn định hơn.")
n_samples = 100
# Tải mô hình
with st.spinner("Đang tải các mô hình..."):
    resnet, flss = load_models()

# Tải dữ liệu CIFAR-10
def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return cifar10

cifar10 = load_cifar10()
class_names = cifar10.classes

# Chọn ảnh từ CIFAR-10
st.subheader("Chọn một ảnh từ tập Test CIFAR-10")
image_idx = st.slider("Chỉ số ảnh", 0, len(cifar10) - 1, 0)
image, label = cifar10[image_idx]
image = image.unsqueeze(0)  # Thêm chiều batch
true_label = class_names[label]

# Chuyển ảnh lên device ngay lập tức
image = image.to(device)

# Hiển thị ảnh gốc và ảnh đối kháng trong một cột
st.subheader("Ảnh gốc và Ảnh đối kháng")
col1, col2 = st.columns(2)
with col1:
    # Chuyển về CPU để hiển thị
    display_image = image.squeeze().cpu().permute(1, 2, 0).numpy()
    st.image(display_image, caption=f"Ảnh gốc (Nhãn thật: {true_label})", use_column_width=True)

# Tạo ảnh đối kháng
with st.spinner("Đang tạo ảnh đối kháng..."):
    label_tensor = torch.tensor([label]).to(device) # Chuyển nhãn lên device
    # Pass alpha (step size) as well
    alpha = epsilon / steps if steps > 0 else 0 # Một cách ước lượng alpha
    adv_image = generate_adversarial_image(resnet, image, label_tensor, epsilon=epsilon, alpha=alpha, steps=steps)

with col2:
    # Chuyển về CPU để hiển thị
    display_adv_image = adv_image.squeeze().cpu().permute(1, 2, 0).numpy()
    st.image(display_adv_image, caption="Ảnh đối kháng (PGD Attack)", use_column_width=True)

# Dự đoán với ResNet-18
st.subheader("Dự đoán của ResNet-18 (Mô hình Baseline)")
try:
    with torch.no_grad(): # Không tính gradient
        # Cả hai ảnh đều đã ở trên device
        resnet_pred_orig = resnet(image).softmax(dim=1)
        resnet_pred_adv = resnet(adv_image).softmax(dim=1)
        
        orig_class = class_names[resnet_pred_orig.argmax().item()]
        orig_conf = resnet_pred_orig.max().item()
        adv_class = class_names[resnet_pred_adv.argmax().item()]
        adv_conf = resnet_pred_adv.max().item()
        
        col_res1, col_res2 = st.columns(2)
        # with col_res1:
        #     st.metric("Dự đoán ảnh gốc", orig_class, f"Độ tin cậy: {orig_conf:.3f}")
        # with col_res2:
            # Highlight nếu ResNet bị đánh lừa
        delta_adv = adv_class != true_label
        st.metric("Dự đoán ảnh đối kháng", adv_class, f"Độ tin cậy: {adv_conf:.3f}", delta_color="inverse" if delta_adv else "normal")
        if delta_adv:
            st.warning("ResNet-18 bị đánh lừa!")
        else:
            st.success("ResNet-18 dự đoán đúng ảnh đối kháng!")
            
except Exception as e:
    st.error(f"Lỗi khi dự đoán với ResNet-18: {e}")
    orig_class, orig_conf = "Lỗi", 0.0
    adv_class, adv_conf = "Lỗi", 0.0

# Dự đoán với FLSS
st.subheader("Dự đoán của FLSS (Stochastic Smoothing)")

try:
    with st.spinner("Đang chạy phân tích FLSS..."):
        latent_fig, pred_fig, flss_result = run_flss_demo(flss, adv_image, n_samples, threshold)

    col_flss1, col_flss2 = st.columns(2)
    with col_flss1:
        st.write("#### Đám mây đặc trưng trong không gian tiềm ẩn")
        if latent_fig:
            st.pyplot(latent_fig)
        else:
            st.info("Không có sẵn biểu đồ không gian tiềm ẩn.")

    with col_flss2:
        st.write("#### Phân phối dự đoán của các mẫu")
        st.pyplot(pred_fig)

    st.info(flss_result)
    
except Exception as e:
    st.error(f"Lỗi khi phân tích FLSS: {e}")
    flss_result = "Phân tích FLSS thất bại"

# Bảng so sánh
st.subheader("Tóm tắt so sánh")

try:
    # Trích xuất kết quả FLSS
    flss_lines = flss_result.split("\n")
    flss_pred = flss_lines[0].split(": ")[1] if len(flss_lines) > 0 else "Lỗi"
    flss_conf = flss_lines[1].split(": ")[1] if len(flss_lines) > 1 else "0.00"
    flss_decision = flss_lines[3].split(": ")[1] if len(flss_lines) > 3 else "Lỗi"

    # Xác định kết quả
    resnet_adv_result = "ĐÚNG" if adv_class == true_label else "SAI"
    flss_final_status = "ĐÚNG" if ("ACCEPT" in flss_decision and flss_pred == true_label) else \
                        ("AN TOÀN (TỪ CHỐI)" if "REJECT" in flss_decision else "SAI")

    comparison_data = {
        "Tiêu chí": ["Loại mô hình", "Dự đoán trên ảnh đối kháng", "Độ tin cậy dự đoán", "Kết quả"],
        "ResNet-18": [
            "Xác định (Deterministic)",
            adv_class,
            f"{adv_conf:.3f}",
            resnet_adv_result
        ],
        "FLSS": [
            "Ngẫu nhiên (Stochastic)",
            flss_pred,
            flss_conf,
            flss_final_status
        ]
    }

    st.table(comparison_data)

    # Tóm tắt hiệu suất
    st.subheader("Nhận định chính")
    if resnet_adv_result == "SAI" and flss_final_status in ["ĐÚNG", "AN TOÀN (TỪ CHỐI)"]:
        st.success("✅ FLSS đã phòng thủ thành công trước cuộc tấn công đối kháng!")
    elif resnet_adv_result == "ĐÚNG" and flss_final_status == "ĐÚNG":
        st.info("ℹ️ Cả hai mô hình đều phân loại đúng ảnh đối kháng.")
    else:
        st.warning("⚠️ Cả hai mô hình đều bị đánh lừa bởi ảnh đối kháng.")

    st.write("**Ưu điểm của FLSS:**")
    st.write("- Dự đoán ngẫu nhiên cung cấp khả năng lượng hóa sự không chắc chắn.")
    st.write("- Cơ chế từ chối có thể lọc ra các dự đoán không chắc chắn.")
    st.write("- Làm mượt ở cấp độ đặc trưng giúp cải thiện tính mạnh mẽ.")
    
except Exception as e:
    st.error(f"Lỗi khi tạo bảng so sánh: {e}")

st.markdown("---")
st.markdown("Ứng dụng demo được phát triển để minh họa FLSS vs ResNet-18 trên CIFAR-10.")