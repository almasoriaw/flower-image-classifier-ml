import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from src import model_configuration as mc
from utils.get_input_args import get_input_args

class Predict:
    def __init__(self, category_names, top_k, checkpoint, image_path, training_compute):
        self.category_names = category_names
        self.top_k = top_k
        self.checkpoint = checkpoint
        self.model, self.model_config = self.load_checkpoint(checkpoint, training_compute)
        self.image_path = image_path

    def load_checkpoint(self, checkpoint_path, training_compute):
        checkpoint = torch.load(checkpoint_path)
        model_config = mc.ModelConfiguration(
            freeze_parameters=checkpoint['freeze_parameters'],
            model_name=checkpoint['model_name'], 
            learning_rate=checkpoint['learning_rate'], 
            hidden_units=checkpoint['hidden_units'], 
            dropout=checkpoint['dropout'], 
            training_compute=training_compute
        )
        model, _, _ = model_config.get_model_and_optimizer()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint.get('class_to_idx', {})
        print(f"Checkpoint loaded from {checkpoint_path}, epoch {checkpoint['epoch']}")
        return model, model_config

    @staticmethod
    def process_image(image_path):
        pil_image = Image.open(image_path)
        
        # Resize the image
        pil_image = pil_image.resize((256, 256))
        
        # Center crop the image
        width, height = pil_image.size
        new_width, new_height = 224, 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        pil_image = pil_image.crop((left, top, right, bottom))
        
        # Convert to numpy array
        np_image = np.array(pil_image)
        
        # Normalize the image
        np_image = np_image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        
        # Reorder dimensions
        np_image = np_image.transpose((2, 0, 1))

        return np_image

    @staticmethod
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        if title:
            ax.set_title(title)
        
        return ax

    def predict(self):
        
        # Process the image
        np_image = self.process_image(self.image_path)
        
        # Convert to PyTorch tensor
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
            
        # Move model to the same device as the tensor
        tensor_image = tensor_image.to(self.model_config.device)
        
        # Set model to evaluation mode and make predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(tensor_image)
        
        # Convert output to probabilities
        ps = torch.exp(output)
        
        # Get the top k probabilities and classes
        top_ps, top_classes = ps.topk(self.top_k, dim=1)
        
        # Convert to lists
        top_ps = top_ps.cpu().numpy().flatten()
        top_classes = top_classes.cpu().numpy().flatten() #Move tensor from GPU to CPU
        
        # Map indices to classes
        class_names = list(self.model.class_to_idx.keys())
        top_classes = [class_names[idx] for idx in top_classes] 
        
        return top_ps, top_classes



if __name__ == "__main__":
    print('===================== Prediction Started! =====================')
    in_arg = get_input_args()
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    image_path = in_arg.image_path
    predictor = Predict(
        cat_to_name,
        in_arg.top_k,
        in_arg.checkpoint,
        in_arg.image_path,
        in_arg.training_compute
    )

    # Predict the image class
    probs, classes = predictor.predict()
    class_names = [cat_to_name[name] for name in classes]

    # Display the image and prediction
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    image = Predict.process_image(image_path)
    Predict.imshow(image, ax=ax1, title=class_names[0])

    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probs, align='center')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()

    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Class Predictions')

    plt.tight_layout()
    plt.show()
    print('===================== Prediction completed! =====================')