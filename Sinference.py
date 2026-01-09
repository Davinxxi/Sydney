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


# Synthetic Data를 평가하는 파일

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
        #device = 'cpu'
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
        self.args['hyparam']['GPGPU']['device']=device

        return device
    
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
        
        self.args['model']['CRN']['input_cnn_channel'] = 1
        self.model=model_dir.get_model_for_doa(self.args['model'], self.args['model_scl'], self.args['hyparam']).to(self.device)

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
  

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']

        self.model_select()
        
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']
        

    def save_output(self, DB_type):
        try:
            now_dict=self.save_config_dict[DB_type]
        except:
            now_dict=self.save_config_dict[int(DB_type)]
            DB_type=int(DB_type)
        
        with open(self.result_folder['inference_folder']+'/'+DB_type+'/result.txt', 'w') as f:
            f.write('\nargmax_acc\n\n')

            k=(now_dict['argmax_acc']/now_dict['number_of_degrees'])
           
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nargmax_doa_error\n\n')
            k=(now_dict['argmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
            
            
          
            f.write('\n\n')

            f.write('\nsoftmax_acc\n\n')

            k=(now_dict['softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nsoftmax_doa_error\n\n')
            k=(now_dict['softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
        
            f.write('\n\n')


            f.write('\nhalf_softmax_acc\n\n')

            k=(now_dict['half_softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nhalf_softmax_doa_error\n\n')
            k=(now_dict['half_softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')

    
    def error_update(self, DB_type, argmax_acc, softmax_acc, half_softmax_acc,argmax_doa_error, softmax_doa_error, half_softmax_doa_error,num):
        now_dict=self.save_config_dict[DB_type]
        
        now_dict['argmax_acc']+=argmax_acc
        now_dict['softmax_acc']+=softmax_acc
        now_dict['half_softmax_acc']+=half_softmax_acc
     
        now_dict['argmax_doa_error']+=argmax_doa_error
        now_dict['softmax_doa_error']+=softmax_doa_error
        now_dict['half_softmax_doa_error']+=half_softmax_doa_error

        now_dict['number_of_degrees']+=num
        self.save_config_dict[DB_type]=now_dict
  
    def config(self,):
        from copy import deepcopy

        self.save_config_dict=dict()

        metric_data={}
        metric_data['argmax_acc']=0
        metric_data['argmax_doa_error']=0
        

        metric_data['softmax_acc']=0
        metric_data['softmax_doa_error']=0

        metric_data['half_softmax_acc']=0
        metric_data['half_softmax_doa_error']=0



        metric_data['number_of_degrees']=0
        
        for room_type in self.result_folder['room_type']:
            os.makedirs(self.result_folder['inference_folder']+room_type, exist_ok=True)

            
            self.save_config_dict[room_type]=deepcopy(metric_data)

   

        return self.args
   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args = args

    def config(self):
        # 필수 경로/플래그
        loader = self.args.setdefault('dataloader', {}).setdefault('test', {}).setdefault('loader', {})
        loader.setdefault('pkl_dir', './SSL_src/prepared/pkl/doa/')
        self.test_loader = Synth_dataload(loader)

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

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()

    
    def run(self, ):
          
        self.test_SD(0)

    
    def plot_out_target(self, out, target, pkl_idx):

        ## save as png
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(out, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.xlabel('Time frame')
        plt.ylabel('Source angle')
        plt.title('Estimated DOA spatial spectrum')
        plt.subplot(2,1,2)
        plt.imshow(target, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.xlabel('Time frame')
        plt.ylabel('Source angle')
        plt.title('Target DOA spatial spectrum')
        os.makedirs(self.logger.result_folder['inference_folder'] + '/pngs/' , exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.logger.result_folder['inference_folder'] + '/pngs/' + pkl_idx.split('/')[-1].replace('.pkl', '.png'), dpi=600)
        plt.close()




    def test_SD(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():

                for iter_num, (mixed, vad, speech_azi, white_snr, coherent_snr, rt60) in enumerate(
                    tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):

                    #mixed=mixed.squeeze(1)
                    mixed=mixed[0].to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)
                    if iter_num == 0:
                        print("mixed.shape (batch):", tuple(mixed.shape))  # (1, 4, 64000)
                        print("vad.shape:", tuple(vad.shape)) # (1, 2, 64000)
                        print("speech_azi.shape:", tuple(speech_azi.shape)) #(1, 2)

                    #out, target, vad_block, emb  = self.model(mixed, vad, speech_azi)    # (B, 3, 360, n), (B, num_spk, n)
                    #out, target, vad_block, embedding  = self.model(mixed, vad, speech_azi)    # (B, 3, 360, n), (B, num_spk, n)
                    # --- 모델 호출
                    ret = self.model(mixed, vad, speech_azi)

                    # --- 가변 반환 안전 처리 (3개 or 4개)
                    if isinstance(ret, tuple):
                        if len(ret) == 4:
                            out, target, vad_block, embedding = ret
                        elif len(ret) == 3:
                            out, target, vad_block = ret
                            embedding = None
                        else:
                            raise ValueError(f"Model returned {len(ret)} items, expected 3 or 4")
                    else:
                        raise TypeError(f"Model should return a tuple, got {type(ret)}")

                    # --- 임베딩 저장 (embedding이 있을 때만)
                    if embedding is not None:
                        pkl_idx = self.dataloader.test_loader.dataset.pkl_list[iter_num]
                        output_name = pkl_idx.split('/')[-1].replace('.pkl', '.npy')
                        embedding_dir = '/root/clssl/SSL_src/prepared/scl_embedding_train/'
                        os.makedirs(embedding_dir, exist_ok=True)

                        # 텐서/넘파이 모두 대응
                        if torch.is_tensor(embedding):
                            arr = embedding.detach().cpu().numpy()
                        elif isinstance(embedding, np.ndarray):
                            arr = embedding
                        else:
                            arr = None

                        if arr is not None:
                            np.save(os.path.join(embedding_dir, output_name), arr)

                    # --- 이후 계산/시각화
                    out = out.sigmoid().detach().cpu()
                    target = target.cpu()
                    speech_azi = speech_azi.cpu()
                    vad_block = vad_block.cpu()
                    num_spk = vad_block.sum(axis=1).max().item()

                    # pkl 경로 구성 (파일명만 있는 pkl_list를 안전하게 절대경로로)
                    pkl_name = self.dataloader.test_loader.dataset.pkl_list[iter_num]
                    pkl_root = self.dataloader.test_loader.dataset.pkl_dir  # synth_data_loader에서 설정한 경로
                    full_pkl_path = os.path.join(pkl_root, pkl_name)

                    # 부모 폴더명 안전 획득 (없으면 'root')
                    parent_dir = os.path.basename(os.path.dirname(full_pkl_path)) or "root"

                    # --- (선택) 임베딩 저장: embedding이 있을 때만
                    if embedding is not None:
                        embedding_dir = '/root/clssl/SSL_src/prepared/scl_embedding_train/'
                        os.makedirs(embedding_dir, exist_ok=True)

                        if torch.is_tensor(embedding):
                            arr = embedding.detach().cpu().numpy()
                        elif isinstance(embedding, np.ndarray):
                            arr = embedding
                        else:
                            arr = None

                        if arr is not None:
                            np.save(os.path.join(embedding_dir, pkl_name.replace('.pkl', '.npy')), arr)

                    # --- 혼합 wav 저장 (부모 폴더명 포함해 정리)
                    import soundfile as sf
                    wav_dir = os.path.join('/root/clssl/Synth/wavfolder/', parent_dir)
                    os.makedirs(wav_dir, exist_ok=True)
                    sf.write(os.path.join(wav_dir, pkl_name.replace('.pkl', '.wav')),
                            mixed.squeeze(0)[0].detach().cpu().numpy(), 16000)


                    (total_argmax_acc, 
                     total_softmax_acc, 
                     total_half_softmax_acc, 
                     total_argmax_doa_error, 
                     total_softmax_doa_error,
                     total_half_softmax_doa_error, 
                     number_of_degrees_to_estimate) = metric.mae.calc_mae(out, target, vad_block, num_spk, speech_azi,\
                                                                        calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                                                                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                                                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])


                    self.logger.error_update(room_type, total_argmax_acc, total_softmax_acc,total_half_softmax_acc, 
                                             total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error,
                                             number_of_degrees_to_estimate)

                    
                    self.learner.memory_delete([mixed, vad, speech_azi, out, target, vad_block, total_argmax_acc, total_softmax_acc, total_half_softmax_acc,
                                             total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error, number_of_degrees_to_estimate])
                  
                self.logger.save_output(room_type)

            break


if __name__=='__main__':
    args=sys.argv[1:]


    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_doa.yaml', 
            'model_scl ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/test.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()