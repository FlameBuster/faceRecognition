from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
mtcnn=MTCNN(image_size=240,margin=0,min_face_size=20)
resnet=InceptionResnetV1(pretrained='vggface2').eval()
dataset=datasets.Imagefolder('/Users/buster/Python/gfg/faceRecognition/photos')
idx_to_class={i:c for c,i in dataset.class_to_idx.items()}

def collate_fn(x):
  return x[0]
loader =DataLoader(dataset,collate_fn=-collate_fn)
face_list=[]
name_list=[]
embedding_list=[]
for img,idx in loader:
  face,prob=mtcnn(img,return_prob=True)
  if face is not None and prob>0.90:
    emb=resnet(face.unsqueeze(0))
    embedding_list.append(emb.detach())
    name_list.append(idx_to_class[idx])

data=[embedding_list,name_list]
torch.save(data,'data.pt')

def face_match(img_path,data_path):
  img=Image.open(img_path)
  face,prob=mtcnn(img,return_prob=True)
  emb=resnet(face.unsqueeze(0).detach())

  saved_data=torch.load('data.pt')
  embedding_list=saved_data[0]
  name_list=saved_data[1]
  dist_list=[]

  for idx,emb_db in enumerate(embedding_list):
    dist=torch.dist(emb,emb_db).item()
    dist_list.append(dist)

    idx_min=dist_list.index(min(dist_list))
    return name_list[idx_min]

person =face_match('1.jpg','data.pt')
print('Identity matched with:',person)
