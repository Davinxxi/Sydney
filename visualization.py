import sys, os
import util
import torch
import numpy as np
import random
import importlib
from tqdm import tqdm
from dataloader.wrap_dataload import Synth_dataload, Real_dataload
import matplotlib.pyplot as plt
import metric
from sklearn.manifold import TSNE
import seaborn as sns


class Hyparam_set():
    
    def __init__(self, args):
        self.args=args

    def set_torch_method(self,):
        try:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=False) # spawn
        except:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=True) # spawn
        

    def randomseed_init(self,):
        np.random.seed(self.args['hyparam']['randomseed'])
        random.seed(self.args['hyparam']['randomseed'])
        torch.manual_seed(self.args['hyparam']['randomseed'])

        
        device = 'cpu'
        self.args['hyparam']['GPGPU']['device']=device
  
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

        #     device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
        #     device= 'cuda'+':'+str(device_primary_num)
        # else:
        #     device= 'cpu'
        # self.args['hyparam']['GPGPU']['device']=device

        # return device
    
    def set_on(self):
        self.set_torch_method()
        self.device=self.randomseed_init()
       
        return self.args


class Learner_config():
    def __init__(self, args) -> None:
        self.args=args
    
    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):

        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'       
        
        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model_for_scl(self.args['model']).to(self.device)

        self.args['hyparam']['model_for_finetune'] = "./results/0808_meanNegDot/model_checkpoint/last_model.tar"

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])   
        

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']

        self.model_select()
        
        return self.args

   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    def config(self):

        self.args['dataloader']['train']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['train']['dataloader_dict']['num_workers'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['num_workers'] = 1
        self.args['dataloader']['val']['loader']['pkl_dir'] = './SSL_src/prepared/pkl/scl/'
        
        self.test_loader=Synth_dataload(self.args['dataloader']['val']['loader'])
        
        return self.args  
    

class Tester():

    def __init__(self, args):

        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()
        self.model=self.learner.model


        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()


    
    def run(self, ):
      
        self.test(0)


    def test(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type = str(room_type)
            self.dataloader.test_loader.dataset.room_type = str(room_type)

            with torch.no_grad():
                embeddings_list = []
                speech_azi_list = []

                for iter_num, (mixed, vad, speech_azi, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):
                    mixed = mixed.to(self.hyperparameter.device)
                    vad = vad.to(self.hyperparameter.device)
                    speech_azi = speech_azi.to(self.hyperparameter.device)

                    out, embedding, speech_azi, vad_block = self.model(mixed, vad, speech_azi)

                    speech_azi = speech_azi.cpu().numpy().flatten()  # (8, )
                    embedding = embedding.cpu().numpy()  # (8, 256, 20)
                    vad_block = vad_block.cpu().numpy()  # (8, 1, 20)

                    embedding_per_frame = np.transpose(embedding, (0, 2, 1)).reshape(-1, embedding.shape[1])  # (8, 20, 256) -> (8*20, 256)
                    speech_azi_repeated = np.repeat(speech_azi, embedding.shape[-1])  # (8*20,)

                    vad_block_flat = vad_block.reshape(-1)  # (8*20,)
                    speech_azi_repeated[vad_block_flat == 0] = 360
                    speech_azi_repeated[speech_azi_repeated == 360] = 1000

                    embeddings_list.append(embedding_per_frame)
                    speech_azi_list.extend(speech_azi_repeated)

                embeddings_array = np.vstack(embeddings_list)
                speech_azi_array = np.array(speech_azi_list)

                # --- Anchor-Positive/Negative Pair 거리 계산 및 시각화 ---
                # anchor: 모든 샘플, positive: 같은 azi, negative: 다른 azi
                from sklearn.metrics import pairwise_distances
                pairwise_dists = pairwise_distances(embeddings_array, embeddings_array, metric='euclidean')
                label_matrix = speech_azi_array[:, None] == speech_azi_array[None, :]
                np.fill_diagonal(label_matrix, False)  # 자기 자신 제외

                # Positive/Negative 쌍 인덱스
                pos_anchor, pos_pair = np.where(label_matrix)
                neg_anchor, neg_pair = np.where(~label_matrix)

                # Positive/Negative 거리
                pos_dists = pairwise_dists[pos_anchor, pos_pair]
                neg_dists = pairwise_dists[neg_anchor, neg_pair]

                # 샘플 수가 많으므로 일부만 사용 (랜덤 샘플링)
                max_pairs = 10000
                if len(pos_dists) > max_pairs:
                    idx = np.random.choice(len(pos_dists), max_pairs, replace=False)
                    pos_dists = pos_dists[idx]
                if len(neg_dists) > max_pairs:
                    idx = np.random.choice(len(neg_dists), max_pairs, replace=False)
                    neg_dists = neg_dists[idx]

                # 거리 분포 히스토그램
                plt.figure(figsize=(8, 5))
                plt.hist(pos_dists, bins=50, alpha=0.6, label='Anchor-Positive', color='g')
                plt.hist(neg_dists, bins=50, alpha=0.6, label='Anchor-Negative', color='r')
                plt.legend()
                plt.title('Embedding Distance Distribution (MeanNegDot_Loss)')
                plt.xlabel('Euclidean Distance')
                plt.ylabel('Count')
                os.makedirs("./results/visualization", exist_ok=True)
                plt.savefig("./results/visualization/distance_hist.png")
                plt.close()

                # --- t-SNE 시각화 및 anchor-positive/negative 연결선 ---
                selected_classes = np.array([0, 20, 60, 180, 300])
                mask = np.isin(speech_azi_array, selected_classes)
                filtered_embeddings = embeddings_array[mask]
                filtered_speech_azi = speech_azi_array[mask]

                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                embeddings_2d = tsne.fit_transform(filtered_embeddings)

                palette = sns.color_palette("hsv", len(selected_classes))
                color_map = {azi: palette[i] for i, azi in enumerate(selected_classes)}
                colors = [color_map[azi] for azi in filtered_speech_azi]

                plt.figure(figsize=(10, 8))
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=10)

                # anchor-positive/negative 연결선 (랜덤 일부만)
                filtered_label_matrix = filtered_speech_azi[:, None] == filtered_speech_azi[None, :]
                np.fill_diagonal(filtered_label_matrix, False)
                pos_anchor, pos_pair = np.where(filtered_label_matrix)
                neg_anchor, neg_pair = np.where(~filtered_label_matrix)
                # 랜덤 샘플링
                max_lines = 300
                if len(pos_anchor) > max_lines:
                    idx = np.random.choice(len(pos_anchor), max_lines, replace=False)
                    pos_anchor = pos_anchor[idx]
                    pos_pair = pos_pair[idx]
                if len(neg_anchor) > max_lines:
                    idx = np.random.choice(len(neg_anchor), max_lines, replace=False)
                    neg_anchor = neg_anchor[idx]
                    neg_pair = neg_pair[idx]
                # 초록: positive, 빨강: negative
                for a, p in zip(pos_anchor, pos_pair):
                    plt.plot([embeddings_2d[a, 0], embeddings_2d[p, 0]], [embeddings_2d[a, 1], embeddings_2d[p, 1]], 'g-', alpha=0.1)
                for a, n in zip(neg_anchor, neg_pair):
                    plt.plot([embeddings_2d[a, 0], embeddings_2d[n, 0]], [embeddings_2d[a, 1], embeddings_2d[n, 1]], 'r-', alpha=0.05)

                legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[azi], markersize=6)
                                for azi in selected_classes]
                plt.legend(legend_handles, [str(azi) for azi in selected_classes], title="Speech Azi", loc='best')

                plt.title("t-SNE Visualization with Anchor-Positive (green) and Anchor-Negative (red) pairs\n(MeanNegDot_Loss)")
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                plt.savefig("./results/visualization/t-SNE_pairs.png")
                plt.close()
                    

if __name__=='__main__':
    args=sys.argv[1:]


    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/test.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()