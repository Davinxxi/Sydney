import sys, os
import util
import torch
import numpy as np
import random
import importlib
from tqdm import tqdm
from dataloader.wrap_dataload import Synth_dataload, Real_dataload
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


class SphereVisualizer():
    
    def __init__(self, args):
        self.args = args
        self.setup_model()
        self.setup_dataloader()
        
    def setup_model(self):
        # Hyparam setup
        np.random.seed(self.args['hyparam']['randomseed'])
        random.seed(self.args['hyparam']['randomseed'])
        torch.manual_seed(self.args['hyparam']['randomseed'])
        
        device = 'cpu'
        self.device = device
        
        # Model setup
        model_name = self.args['model']['name']
        model_import = 'models.' + model_name + '.main'
        model_dir = importlib.import_module(model_import)
        
        self.model = model_dir.get_model_for_scl(self.args['model']).to(self.device)
        
        if self.args['hyparam']['finetune']:
            # 8/26 visualization에서 train.yaml을 한번에 이용하기 위함
            
            #print(f"Loading model weights from: {self.args['hyparam']['model_used_for_scl']}")
            #trained = torch.load(self.args['hyparam']['model_used_for_scl'], map_location=self.device)
            print(f"Loading model weights from: {self.args['hyparam']['trained_scl_model_path']}")
            trained = torch.load(self.args['hyparam']['trained_scl_model_path'], map_location=self.device)

            # 모델 가중치 로딩 상세 정보
            missing_keys, unexpected_keys = self.model.load_state_dict(trained['model_state_dict'], strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print("Some missing keys found - model may not be fully loaded!")
                print(f"First 5 missing keys: {missing_keys[:5]}")
        else:
            print("Warning: Using randomly initialized model weights!")
        
        self.model.eval()
        
    def setup_dataloader(self):
        self.args['dataloader']['val']['loader']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['num_workers'] = 1
        self.args['dataloader']['val']['loader']['pkl_dir'] = './SSL_src/prepared/pkl/scl/'
        
        self.test_loader = Synth_dataload(self.args['dataloader']['val']['loader'])
        self.test_loader.dataset.room_type = 'evaluation'
        
    def collect_embeddings(self, max_samples=50):
        """임베딩 수집"""
        
        embeddings_list = []
        azimuths_list = []
        
        print("Collecting embeddings for sphere visualization...")
        with torch.no_grad():
            for iter_num, (mixed, vad, speech_azi, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.test_loader)):
                if iter_num >= max_samples:
                    break
                    
                mixed = mixed.to(self.device)
                vad = vad.to(self.device)
                speech_azi = speech_azi.to(self.device)
                
                out, embedding, speech_azi, vad_block = self.model(mixed, vad, speech_azi)
                
                speech_azi = speech_azi.cpu().numpy().flatten()
                embedding = embedding.cpu().numpy()  # (8, 256/512, 20)
                vad_block = vad_block.cpu().numpy()  # (8, 1, 20)
                
                # 각 시간 프레임별로 분석
                for frame_idx in range(embedding.shape[-1]):
                    frame_embeddings = embedding[:, :, frame_idx]  # (8, 256/512)
                    frame_vad = vad_block[:, 0, frame_idx]  # (8,)
                    
                    valid_indices = frame_vad == 1
                    if np.any(valid_indices):
                        embeddings_list.append(frame_embeddings[valid_indices])
                        azimuths_list.extend(speech_azi[valid_indices])
        
        # 데이터 결합
        all_embeddings = np.vstack(embeddings_list)
        all_azimuths = np.array(azimuths_list)
        
        print(f"Total samples: {len(all_embeddings)}")
        print(f"Embedding shape: {all_embeddings.shape}")
        print(f"Azimuth range: {all_azimuths.min():.1f} - {all_azimuths.max():.1f}")
        
        return all_embeddings, all_azimuths
    
    def project_to_3d_sphere(self, embeddings, method='pca'):
        """고차원 임베딩을 3D unit sphere로 투영"""
        
        print(f"Projecting {embeddings.shape[1]}D embeddings to 3D sphere using {method}...")
        
        if method == 'pca':
            # PCA로 3차원으로 축소
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            
        elif method == 'random':
            # 랜덤 투영
            np.random.seed(42)
            projection_matrix = np.random.randn(embeddings.shape[1], 3)
            embeddings_3d = embeddings @ projection_matrix
            
        elif method == 'first3':
            # 처음 3차원만 사용
            embeddings_3d = embeddings[:, :3]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Unit sphere로 정규화
        norms = np.linalg.norm(embeddings_3d, axis=1, keepdims=True)
        embeddings_3d_normalized = embeddings_3d / (norms + 1e-8)
        
        return embeddings_3d_normalized
    
    def create_sphere_wireframe(self, resolution=20):
        """구 와이어프레임 생성"""
        
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        return x, y, z
    
    def visualize_matplotlib_3d(self, embeddings_3d, azimuths, save_path='./results/sphere_viz/matplotlib_sphere.png'):
        """Matplotlib로 3D 구면 시각화"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 샘플링 (너무 많으면 시각화가 어려움)
        if len(embeddings_3d) > 1000:
            indices = np.random.choice(len(embeddings_3d), 1000, replace=False)
            embeddings_3d = embeddings_3d[indices]
            azimuths = azimuths[indices]
        
        fig = plt.figure(figsize=(15, 5))
        
        # 구 와이어프레임
        sphere_x, sphere_y, sphere_z = self.create_sphere_wireframe()
        
        # 1. 전체 뷰
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
        scatter1 = ax1.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                              c=azimuths, cmap='hsv', s=20, alpha=0.8)
        ax1.set_title('3D Unit Sphere - Full View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Azimuth (degrees)')
        
        # 2. 위에서 본 뷰 (XY plane)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
        scatter2 = ax2.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                              c=azimuths, cmap='hsv', s=20, alpha=0.8)
        ax2.view_init(elev=90, azim=0)  # 위에서 보기
        ax2.set_title('Top View (XY plane)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 3. 옆에서 본 뷰 (XZ plane)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
        scatter3 = ax3.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                              c=azimuths, cmap='hsv', s=20, alpha=0.8)
        ax3.view_init(elev=0, azim=0)  # 옆에서 보기
        ax3.set_title('Side View (XZ plane)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matplotlib 3D visualization saved to: {save_path}")
    
    def visualize_plotly_interactive(self, embeddings_3d, azimuths, save_path='./results/sphere_viz/interactive_sphere.html'):
        """Plotly로 인터랙티브 3D 구면 시각화"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 샘플링
        if len(embeddings_3d) > 2000:
            indices = np.random.choice(len(embeddings_3d), 2000, replace=False)
            embeddings_3d = embeddings_3d[indices]
            azimuths = azimuths[indices]
        
        # 구 표면 생성
        sphere_x, sphere_y, sphere_z = self.create_sphere_wireframe(resolution=30)
        
        fig = go.Figure()
        
        # 구 와이어프레임 추가
        for i in range(sphere_x.shape[0]):
            fig.add_trace(go.Scatter3d(
                x=sphere_x[i], y=sphere_y[i], z=sphere_z[i],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for j in range(sphere_x.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=sphere_x[:, j], y=sphere_y[:, j], z=sphere_z[:, j],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 임베딩 포인트 추가
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=azimuths,
                colorscale='HSV',
                colorbar=dict(title='Azimuth (degrees)'),
                showscale=True
            ),
            text=[f'Azimuth: {azi:.1f}°' for azi in azimuths],
            hovertemplate='<b>Azimuth: %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            name='Embeddings'
        ))
        
        fig.update_layout(
            title='Interactive 3D Unit Sphere Visualization of Embeddings',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(save_path)
        print(f"Interactive Plotly visualization saved to: {save_path}")
        
        return fig
    
    def analyze_sphere_distribution(self, embeddings_3d, azimuths):
        """구면 상의 분포 분석"""
        
        print("\n=== Sphere Distribution Analysis ===")
        
        # 구면 좌표계로 변환
        x, y, z = embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2]
        
        # 방위각 (azimuth): atan2(y, x)
        sphere_azimuth = np.degrees(np.arctan2(y, x)) % 360
        
        # 극각 (elevation): arccos(z)
        sphere_elevation = np.degrees(np.arccos(np.clip(z, -1, 1)))
        
        print(f"Sphere azimuth range: {sphere_azimuth.min():.1f} - {sphere_azimuth.max():.1f}")
        print(f"Sphere elevation range: {sphere_elevation.min():.1f} - {sphere_elevation.max():.1f}")
        
        # 실제 방위각과 구면 방위각의 상관관계
        from scipy.stats import pearsonr, spearmanr
        
        corr_pearson, p_pearson = pearsonr(azimuths, sphere_azimuth)
        corr_spearman, p_spearman = spearmanr(azimuths, sphere_azimuth)
        
        print(f"\nCorrelation between true azimuth and sphere azimuth:")
        print(f"  Pearson correlation: {corr_pearson:.4f} (p={p_pearson:.4f})")
        print(f"  Spearman correlation: {corr_spearman:.4f} (p={p_spearman:.4f})")
        
        if abs(corr_pearson) > 0.5:
            print("  -> Good correlation! Embeddings preserve azimuth in sphere projection")
        elif abs(corr_pearson) > 0.2:
            print("  -> Moderate correlation")
        else:
            print("  -> Poor correlation")
        
        return sphere_azimuth, sphere_elevation
    
    def run_sphere_visualization(self, projection_method='pca'):
        """전체 구면 시각화 실행"""
        
        print("Starting 3D sphere visualization...")
        
        # 1. 임베딩 수집
        embeddings, azimuths = self.collect_embeddings()
        
        # 2. 3D 구면으로 투영
        embeddings_3d = self.project_to_3d_sphere(embeddings, method=projection_method)
        
        # 3. 분포 분석
        sphere_azi, sphere_ele = self.analyze_sphere_distribution(embeddings_3d, azimuths)
        
        # 4. Matplotlib 시각화
        self.visualize_matplotlib_3d(embeddings_3d, azimuths)
        
        # 5. Plotly 인터랙티브 시각화
        try:
            self.visualize_plotly_interactive(embeddings_3d, azimuths)
        except Exception as e:
            print(f"Plotly visualization failed: {e}")
            print("You can install plotly with: pip install plotly")
        
        print("\nSphere visualization completed!")


if __name__ == '__main__':
    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/test.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    
    args = util.util.get_yaml_args(args)
    
    visualizer = SphereVisualizer(args)
    
    # PCA로 투영해서 시각화
    visualizer.run_sphere_visualization(projection_method='pca')
    
    # 다른 투영 방법도 시도
    # visualizer.run_sphere_visualization(projection_method='random')
    # visualizer.run_sphere_visualization(projection_method='first3')
