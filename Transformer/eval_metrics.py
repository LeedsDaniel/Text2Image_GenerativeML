from torchmetrics.multimodal.clip_score import CLIPScore

"""!pip install pytorch-lightning
!pip install torchmetrics"""

#Image - Tensor/list of tensors
#Prompt - plain text string/of list of strings
def clip_score(prompt, image):
  metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16”)
  score = metric(image_tensor, prompt)
  return score
  
