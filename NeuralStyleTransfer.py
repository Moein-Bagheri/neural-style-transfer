import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import warnings
import time

warnings.filterwarnings('ignore')

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
if torch.cuda.is_available():
    print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")


class EnhancedVGGFeatureExtractor(nn.Module):
    """Enhanced VGG19-based feature extractor with advanced capabilities"""

    def __init__(self):
        super(EnhancedVGGFeatureExtractor, self).__init__()

        vgg = models.vgg19(pretrained=True).features.to(device)
        vgg.eval()

        # Freeze parameters for stability
        for param in vgg.parameters():
            param.requires_grad_(False)

        self.vgg = vgg

        # Advanced layer mapping with semantic meaning
        self.layer_mapping = {
            '0': 'conv1_1',  # Low-level edges
            '5': 'conv2_1',  # Basic shapes
            '10': 'conv3_1',  # Complex patterns
            '19': 'conv4_1',  # High-level features
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'  # Abstract features
        }

        # Multi-level content and style layers
        self.content_layers = ['conv4_2', 'conv5_1']  # Multiple content layers
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # Adaptive weights for different style layers
        self.style_layer_weights = {
            'conv1_1': 1.0,  # Fine details
            'conv2_1': 0.8,  # Medium patterns
            'conv3_1': 0.6,  # Complex textures
            'conv4_1': 0.4,  # High-level style
            'conv5_1': 0.2  # Abstract style
        }

        # VGG normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    def normalize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor for VGG processing"""
        normalized = tensor.clone()
        for i in range(3):
            normalized[:, i] = (normalized[:, i] - self.mean[i]) / self.std[i]
        return normalized

    def extract_features(self, x: torch.Tensor, target_layers: List[str] = None) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers"""
        if target_layers is None:
            target_layers = self.content_layers + self.style_layers

        features = {}
        x = self.normalize_input(x)

        for name, layer in self.vgg._modules.items():
            x = layer(x)
            layer_name = self.layer_mapping.get(name)
            if layer_name in target_layers:
                features[layer_name] = x

        return features


class AdvancedLossCalculator:
    """loss calculation with multiple sophisticated techniques"""

    @staticmethod
    def enhanced_gram_matrix(tensor: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Enhanced Gram matrix with better normalization"""
        batch_size, channels, height, width = tensor.size()
        features = tensor.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())

        if normalize:
            # Better normalization for stable training
            return gram.div(batch_size * channels * height * width)
        return gram

    @staticmethod
    def multi_scale_content_loss(input_features: Dict[str, torch.Tensor],
                                 target_features: Dict[str, torch.Tensor],
                                 content_layers: List[str]) -> torch.Tensor:
        """Multi-scale content loss for better content preservation"""
        total_loss = 0
        weights = {'conv4_2': 1.0, 'conv5_1': 0.5}  # Different weights for different scales

        for layer in content_layers:
            if layer in input_features and layer in target_features:
                weight = weights.get(layer, 1.0)
                loss = F.mse_loss(input_features[layer], target_features[layer])
                total_loss += weight * loss

        return total_loss

    @classmethod
    def advanced_style_loss(cls, input_features: Dict[str, torch.Tensor],
                            target_features: Dict[str, torch.Tensor],
                            style_layers: List[str],
                            layer_weights: Dict[str, float]) -> torch.Tensor:
        """Advanced style loss with layer weighting"""
        total_loss = 0

        for layer in style_layers:
            if layer in input_features and layer in target_features:
                input_gram = cls.enhanced_gram_matrix(input_features[layer])
                target_gram = cls.enhanced_gram_matrix(target_features[layer])

                layer_loss = F.mse_loss(input_gram, target_gram)
                weight = layer_weights.get(layer, 1.0)
                total_loss += weight * layer_loss

        return total_loss

    @staticmethod
    def adaptive_total_variation_loss(image: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """Adaptive total variation for better smoothing"""
        batch_size, channels, height, width = image.size()

        # Calculate gradients
        tv_h = torch.pow(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), beta).sum()
        tv_w = torch.pow(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), beta).sum()

        # Adaptive normalization based on image size
        normalization = batch_size * channels * height * width
        return (tv_h + tv_w) / normalization

    @staticmethod
    def color_coherence_loss(input_image: torch.Tensor, content_image: torch.Tensor) -> torch.Tensor:
        """Preserve color coherence between input and content"""
        # Color distribution matching
        input_mean = torch.mean(input_image, dim=[2, 3], keepdim=True)
        content_mean = torch.mean(content_image, dim=[2, 3], keepdim=True)

        input_std = torch.std(input_image, dim=[2, 3], keepdim=True)
        content_std = torch.std(content_image, dim=[2, 3], keepdim=True)

        mean_loss = F.mse_loss(input_mean, content_mean)
        std_loss = F.mse_loss(input_std, content_std)

        return mean_loss + 0.5 * std_loss

    @staticmethod
    def perceptual_loss(input_features: Dict[str, torch.Tensor],
                        content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Additional perceptual loss for better quality"""
        loss = 0
        for layer in ['conv3_1', 'conv4_1']:
            if layer in input_features and layer in content_features:
                loss += F.l1_loss(input_features[layer], content_features[layer])
        return loss


class ProfessionalStyleTransfer:
    """Professional-grade style transfer system combining best practices"""

    def __init__(self, image_size: int = 512):
        self.device = device
        self.image_size = image_size

        # Initialize enhanced feature extractor
        self.feature_extractor = EnhancedVGGFeatureExtractor()
        self.loss_calculator = AdvancedLossCalculator()

        # Professional-grade loss weights (empirically optimized)
        self.loss_weights = {
            'content_weight': 0.1,
            'style_weight': 50000.0,  # Reduced for better balance
            'tv_weight': 1.0,  # Moderate smoothing
            'color_weight': 00.0,  # Strong color preservation
            'perceptual_weight': 0.0  # Subtle perceptual enhancement
        }

        # Advanced preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.LANCZOS),  # High-quality resizing
            transforms.ToTensor(),
        ])

        print(f" Professional Style Transfer initialized")
        print(f" Image size: {image_size}x{image_size}")
        print(f" Loss weights: {self.loss_weights}")

    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image with professional quality"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            print(f" Loaded image: {image.size} -> {self.image_size}x{self.image_size}")

            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Ensure valid range [0, 1]
            image_tensor = torch.clamp(image_tensor, 0, 1)

            return image_tensor

        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")

    def multi_stage_optimization(self, content_image: torch.Tensor,
                                 style_image: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """Multi-stage optimization for professional results"""

        print(" Extracting target features...")

        # Extract target features (detached for memory efficiency)
        with torch.no_grad():
            content_features = self.feature_extractor.extract_features(content_image)
            style_features = self.feature_extractor.extract_features(style_image)

        # Professional initialization: content + controlled noise
        input_image = content_image.clone()
        noise = torch.randn_like(input_image) * 0.5  # Subtle noise
        input_image = torch.clamp(input_image + noise, 0, 1)
        input_image.requires_grad_(True)

        # Multi-stage optimization strategy
        optimization_stages = [
            {'name': 'Coarse', 'steps': 200, 'lr': 0.05, 'style_weight': 5000.0},
            {'name': 'Refine', 'steps': 300, 'lr': 0.03, 'style_weight': 10000.0},
            {'name': 'Polish', 'steps': 200, 'lr': 0.01, 'style_weight': 15000.0}
        ]

        history = []
        total_steps = 0

        for stage_idx, stage in enumerate(optimization_stages):
            print(f"\n Stage {stage_idx + 1}: {stage['name']}")
            print(f"   Steps: {stage['steps']}, LR: {stage['lr']}, Style Weight: {stage['style_weight']}")

            # Update style weight for current stage
            current_weights = self.loss_weights.copy()
            current_weights['style_weight'] = stage['style_weight']

            # Optimizer with stage-specific learning rate
            optimizer = optim.LBFGS([input_image], lr=stage['lr'], max_iter=20)

            stage_losses = []

            def closure():
                nonlocal total_steps
                optimizer.zero_grad()

                # Clamp to valid range
                with torch.no_grad():
                    input_image.clamp_(0, 1)

                # Extract current features
                current_features = self.feature_extractor.extract_features(input_image)

                # Calculate all losses
                content_loss = self.loss_calculator.multi_scale_content_loss(
                    current_features, content_features,
                    self.feature_extractor.content_layers
                )

                style_loss = self.loss_calculator.advanced_style_loss(
                    current_features, style_features,
                    self.feature_extractor.style_layers,
                    self.feature_extractor.style_layer_weights
                )

                tv_loss = self.loss_calculator.adaptive_total_variation_loss(input_image)

                color_loss = torch.tensor(0.0).to(input_image.device)

                perceptual_loss = self.loss_calculator.perceptual_loss(
                    current_features, content_features
                )

                # Combined loss
                total_loss = (current_weights['content_weight'] * content_loss +
                              current_weights['style_weight'] * style_loss +
                              current_weights['tv_weight'] * tv_loss +
                              current_weights['color_weight'] * color_loss +
                              current_weights['perceptual_weight'] * perceptual_loss)

                # Store loss information
                loss_info = {
                    'step': total_steps,
                    'total': total_loss.item(),
                    'content': content_loss.item(),
                    'style': style_loss.item(),
                    'tv': tv_loss.item(),
                    'color': color_loss.item(),
                    'perceptual': perceptual_loss.item()
                }
                stage_losses.append(loss_info)

                total_loss.backward(retain_graph=True)
                total_steps += 1

                return total_loss

            # Run optimization for current stage
            for step in range(stage['steps']):
                try:
                    optimizer.step(closure)

                    if step % 50 == 0 and stage_losses:
                        last_loss = stage_losses[-1]
                        print(f"   Step {step:3d}: Total={last_loss['total']:.3f}, "
                              f"Style={last_loss['style']:.3f}, "
                              f"Content={last_loss['content']:.3f}")

                except RuntimeError as e:
                    if "second time" in str(e).lower():
                        print(f"    Stage {stage['name']} completed early at step {step}")
                        break
                    else:
                        raise e

            history.extend(stage_losses)
            print(f"    Stage {stage['name']} completed")

        print(f"\n Multi-stage optimization completed! Total steps: {total_steps}")

        return input_image.detach(), history

    def save_professional_result(self, result_tensor: torch.Tensor,
                                 content_path: str, style_path: str,
                                 output_dir: str = "output") -> str:
        """Save result with professional quality"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        timestamp = int(time.time())
        filename = f"professional_{content_name}_{style_name}_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)

        # Convert tensor to image
        with torch.no_grad():
            result_image = result_tensor.cpu().clone().squeeze(0)
            result_image = torch.clamp(result_image, 0, 1)
            result_image = result_image.permute(1, 2, 0).numpy()
            result_image = (result_image * 255).astype(np.uint8)

        # Save with high quality
        pil_image = Image.fromarray(result_image)
        pil_image.save(output_path, 'JPEG', quality=95, optimize=True)

        print(f" result saved: {output_path}")
        return output_path

    def create_professional_comparison(self, content_path: str, style_path: str,
                                       result_path: str, loss_history: List[Dict]) -> str:
        """Create professional comparison and analysis"""

        fig = plt.figure(figsize=(20, 12))

        # Load images
        content_img = Image.open(content_path)
        style_img = Image.open(style_path)
        result_img = Image.open(result_path)

        # Image comparison (top row)
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(content_img)
        ax1.set_title('Content Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(style_img)
        ax2.set_title('Style Reference', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(result_img)
        ax3.set_title('Result', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # Side-by-side comparison
        ax4 = plt.subplot(2, 4, 4)
        comparison = np.hstack([np.array(content_img.resize((256, 256))),
                                np.array(result_img.resize((256, 256)))])
        ax4.imshow(comparison)
        ax4.set_title('Before → After', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # Loss analysis (bottom row)
        if loss_history:
            steps = [entry['step'] for entry in loss_history]
            total_losses = [entry['total'] for entry in loss_history]
            style_losses = [entry['style'] for entry in loss_history]
            content_losses = [entry['content'] for entry in loss_history]

            ax5 = plt.subplot(2, 4, 5)
            ax5.plot(steps, total_losses, 'b-', linewidth=2, label='Total Loss')
            ax5.set_title('Total Loss Convergence', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Steps')
            ax5.set_ylabel('Loss')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

            ax6 = plt.subplot(2, 4, 6)
            ax6.plot(steps, style_losses, 'r-', linewidth=2, label='Style Loss')
            ax6.plot(steps, content_losses, 'g-', linewidth=2, label='Content Loss')
            ax6.set_title('Style vs Content Loss', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Steps')
            ax6.set_ylabel('Loss')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

            # Loss components breakdown
            ax7 = plt.subplot(2, 4, 7)
            tv_losses = [entry['tv'] for entry in loss_history]
            color_losses = [entry['color'] for entry in loss_history]
            ax7.plot(steps, tv_losses, 'm-', linewidth=2, label='TV Loss')
            ax7.plot(steps, color_losses, 'c-', linewidth=2, label='Color Loss')
            ax7.set_title('Additional Loss Components', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Steps')
            ax7.set_ylabel('Loss')
            ax7.grid(True, alpha=0.3)
            ax7.legend()

            # Final statistics
            ax8 = plt.subplot(2, 4, 8)
            final_losses = loss_history[-1]
            loss_names = ['Content', 'Style', 'TV', 'Color', 'Perceptual']
            loss_values = [final_losses['content'], final_losses['style'],
                           final_losses['tv'], final_losses['color'],
                           final_losses['perceptual']]

            bars = ax8.bar(loss_names, loss_values,
                           color=['green', 'red', 'magenta', 'cyan', 'orange'])
            ax8.set_title('Final Loss Breakdown', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Loss Value')
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, loss_values):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Save comparison
        comparison_path = result_path.replace('.jpg', '_analysis.png')
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()

        print(f" analysis saved: {comparison_path}")
        return comparison_path


def main_professional_demo():
    """Professional style transfer demonstration"""

    print(" NEURAL STYLE TRANSFER")
    print("=" * 50)
    print()

    # Get input paths
    content_path = input(" Content image path: ").strip()
    if not os.path.exists(content_path):
        print(f" Content image not found: {content_path}")
        return

    style_path = input(" Style image path: ").strip()
    if not os.path.exists(style_path):
        print(f" Style image not found: {style_path}")
        return

    # Image size option
    print("\n Select image size:")
    print("1. 512x512 (High quality, slower)")
    print("2. 256x256 (Fast, good quality)")

    size_choice = input("Choose (1-2): ").strip()
    image_size = 512 if size_choice == '1' else 256

    print(f"\n Starting professional style transfer...")
    print(f" Image size: {image_size}x{image_size}")
    print("  This may take 5-15 minutes for best quality...")

    try:
        # Initialize professional system
        style_transfer = ProfessionalStyleTransfer(image_size=image_size)

        # Load and preprocess images
        print("\n Loading images...")
        content_image = style_transfer.load_and_preprocess_image(content_path)
        style_image = style_transfer.load_and_preprocess_image(style_path)

        # Professional optimization
        print("\n Starting multi-stage optimization...")
        start_time = time.time()

        result_image, loss_history = style_transfer.multi_stage_optimization(
            content_image, style_image
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Save professional result
        print("\n Saving result...")
        result_path = style_transfer.save_professional_result(
            result_image, content_path, style_path
        )

        # Create comprehensive analysis
        print("\n Creating analysis...")
        analysis_path = style_transfer.create_professional_comparison(
            content_path, style_path, result_path, loss_history
        )

        # Final summary
        print(f"\n STYLE TRANSFER COMPLETED!")
        print("=" * 50)
        print(f" Processing time: {processing_time:.1f} seconds")
        print(f" Total optimization steps: {len(loss_history)}")
        print(f" Final loss: {loss_history[-1]['total']:.4f}")
        print(f" Result saved: {result_path}")
        print(f" Analysis saved: {analysis_path}")
        print()
        print("   • Advanced loss functions (5 different loss types)")
        print("   • Color coherence preservation")
        print("   • Adaptive total variation smoothing")
        print("   • Comprehensive loss analysis")
        print()

    except Exception as e:
        print(f" Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_professional_demo()