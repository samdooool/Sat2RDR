import os
import numpy as np
import matplotlib.pyplot as plt

def save_images(epoch, input_image, true_image, pred_image, A_path, B_path):
    input_image = input_image.cpu().detach().numpy()
    true_image = true_image.cpu().detach().numpy()
    pred_image = pred_image.cpu().detach().numpy()

    input_ori_path = os.path.basename(A_path)[:-4]
    true_ori_path = os.path.basename(B_path)[:-4]
    np.save(f"./results/{true_ori_path}", pred_image)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap='gray_r')
    plt.title(f'Input Image ({input_ori_path})')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_image, cmap='gray')
    plt.title(f'Ground Truth ({true_ori_path})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_image, cmap='gray')
    plt.title('Predicted Image')
    plt.axis('off')

    # Save the figure to a file
    plt.savefig(f"./results/epoch_{epoch}_results.png")
    plt.close()

    print(f"Saved images for epoch {epoch} as 'epoch_{epoch}_results.png'")
