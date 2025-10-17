import torch
import numpy as np
import matplotlib.pyplot as plt
import os
def SupCon_curve2():
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 저장 경로
    save_path = "/root/clssl/SSL_src_DV/loss/pngs/"
    os.makedirs(save_path, exist_ok=True)

    tau = 0.1
    d = 128   # embedding dimension

    # -------------------------------------------------
    # Theoretical θ-Loss curve (0° ≤ θ ≤ 180°)
    # -------------------------------------------------
    def loss_curve(theta_rad, tau, neg_cos):
        cos_ip = np.cos(theta_rad)  # positive와 anchor 사이 각도 (radian)
        sim_pos = np.exp(cos_ip / tau)
        sim_negs = np.exp(neg_cos / tau) * (7*64)   # 단일 평균 negative 반영
        return -np.log(sim_pos / (sim_pos + sim_negs))

    # θ를 degree로 생성하고 radian으로 변환해서 계산
    thetas_deg = np.linspace(0, 180, 2000)         # 더 촘촘하게
    thetas_rad = np.radians(thetas_deg)            # radian 변환

    # negative 평균 각도 50°
    neg_theta_deg = 4
    neg_cos = np.cos(np.radians(neg_theta_deg))

    # loss 계산
    loss_vals = [loss_curve(th, tau, neg_cos) for th in thetas_rad]
    loss_vals = np.array(loss_vals)

    # Loss=1에 가장 가까운 지점 찾기
    target_loss = 1.0
    idx = np.argmin(np.abs(loss_vals - target_loss))
    theta_at_loss1 = thetas_deg[idx]
    loss_at_theta = loss_vals[idx]

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(thetas_deg, loss_vals, label="Contrastive Loss")
    plt.axhline(1, color="red", linestyle="--", label="Loss=1")
    plt.axvline(theta_at_loss1, color="green", linestyle="--", 
                label=f"θ ≈ {theta_at_loss1:.2f}°")
    plt.scatter([theta_at_loss1], [loss_at_theta], color="black", zorder=5)

    plt.xlabel("θ (degrees)")
    plt.ylabel("Loss")
    plt.title("Theoretical Contrastive Loss Curve (0° ≤ θ ≤ 180°)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "contrastive_loss_curve_with_marker2.png")
    plt.close()

    print(f"Loss=1 at θ ≈ {theta_at_loss1:.2f}°")

def SupCon_curve():
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 저장 경로
    save_path = "/root/clssl/SSL_src_DV/loss/pngs/"
    os.makedirs(save_path, exist_ok=True)

    tau = 0.1
    d = 128   # embedding dimension

    # -------------------------------------------------
    # Theoretical θ-Loss curve (0° ≤ θ ≤ 180°)
    # -------------------------------------------------
    def loss_curve(theta_rad, tau, negatives):
        cos_ip = np.cos(theta_rad)  # positive와 anchor 사이 각도 (radian)
        sim_pos = np.exp(cos_ip / tau)
        sim_negs = np.exp(negatives / tau).sum()
        return -np.log(sim_pos / (sim_pos + sim_negs))

    # θ를 degree로 생성하고 radian으로 변환해서 계산
    thetas_deg = np.linspace(0, 180, 200)         # 0° ~ 180°
    thetas_rad = np.radians(thetas_deg)           # radian 변환

    neg_theta_deg = 50
    neg_cos = np.cos(np.radians(neg_theta_deg))


    loss_vals = [loss_curve(th, tau, neg_cos) for th in thetas_rad]

    plt.figure(figsize=(7,5))
    plt.plot(thetas_deg, loss_vals)
    plt.xlabel("θ (degrees)")
    plt.ylabel("Loss")
    plt.title("Theoretical Contrastive Loss Curve (0° ≤ θ ≤ 180°)")
    plt.tight_layout()
    plt.savefig(save_path + "contrastive_loss_curve_deg_2.png")
    plt.close()

def FlipCon_curve():
    # 저장 경로
    save_path = "/root/clssl/SSL_src_DV/loss/pngs/"
    os.makedirs(save_path, exist_ok=True)

    tau = 0.1   # temperature

    # -------------------------------------------------
    # 변형된 Contrastive Loss 수식 구현
    # -------------------------------------------------
    def loss_curve(theta_rad, tau, negatives):
        cos_ip = np.cos(theta_rad)  # cos(theta)
        sim_pos = np.exp(-cos_ip / tau)   # 분자 exp(-zi·zp/τ)
        sim_negs = np.sum(np.exp(-negatives / tau))  # 분모 합 exp(-zi·za/τ)
        return np.log(sim_pos / (sim_pos + sim_negs))

    # θ: 0° ~ 180°
    thetas_deg = np.linspace(0, 180, 200)
    thetas_rad = np.radians(thetas_deg)

    # 예시: negatives를 90° (cos=0) 근처라 가정
    neg_cos_vals = np.array([0.0])

    loss_vals = [loss_curve(th, tau, neg_cos_vals) for th in thetas_rad]

    plt.figure(figsize=(7,5))
    plt.plot(thetas_deg, loss_vals, label="Modified SupCon Loss")
    plt.xlabel("θ (degrees)")
    plt.ylabel("Loss")
    plt.title("Modified Contrastive Loss Curve (0° ≤ θ ≤ 180°)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "flipcon_curve.png")
    plt.close()

def FlipCon_curve2():
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 저장 경로
    save_path = "/root/clssl/SSL_src_DV/loss/pngs/"
    os.makedirs(save_path, exist_ok=True)

    tau = 0.1   # temperature

    # -------------------------------------------------
    # 변형된 Contrastive Loss 수식 구현
    # -------------------------------------------------
    def loss_curve(theta_rad, tau, negatives):
        cos_ip = np.cos(theta_rad)  # cos(theta)
        sim_pos = np.exp(-cos_ip / tau)   # 분자 exp(-zi·zp/τ)
        sim_negs = np.sum(np.exp(-negatives / tau))*(7*64)  # 분모 합 exp(-zi·za/τ)
        return np.log(sim_pos / (sim_pos + sim_negs))

    # θ: 0° ~ 180°
    thetas_deg = np.linspace(0, 180, 2000)  # 촘촘하게
    thetas_rad = np.radians(thetas_deg)

    # 예시: negatives를 50° → cos(50°)
    neg_theta_deg = 5
    neg_cos_vals = np.array([np.cos(np.radians(neg_theta_deg))])

    # loss 계산
    loss_vals = [loss_curve(th, tau, neg_cos_vals) for th in thetas_rad]
    loss_vals = np.array(loss_vals)

    # Loss = -1과 가장 가까운 지점 찾기
    target_loss = -1.0
    idx = np.argmin(np.abs(loss_vals - target_loss))
    theta_at_loss = thetas_deg[idx]
    loss_at_theta = loss_vals[idx]

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(thetas_deg, loss_vals, label="FlipCon Loss")
    plt.axhline(target_loss, color="red", linestyle="--", label="Loss=-1")
    plt.axvline(theta_at_loss, color="green", linestyle="--", 
                label=f"θ ≈ {theta_at_loss:.2f}°")
    plt.scatter([theta_at_loss], [loss_at_theta], color="black", zorder=5)

    plt.xlabel("θ (degrees)")
    plt.ylabel("Loss")
    plt.title("FlipCon Loss Curve (0° ≤ θ ≤ 180°)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "flipcon_curve_with_marker4.png")
    plt.close()

    print(f"Loss=-1 at θ ≈ {theta_at_loss:.2f}°")


FlipCon_curve2()