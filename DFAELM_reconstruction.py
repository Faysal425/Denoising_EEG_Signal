import torch
import torch.nn as nn

# ========== Define Channel Attention Module ==========
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Global max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared MLP for dimensionality reduction and expansion
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# ========== Define Temporal Attention Module ==========
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, L = x.size()
        
        # [B, C, L] -> [B, C, L]
        query = self.conv1(x)
        key = self.conv2(x)
        value = self.conv3(x)
        
        # Reshape for attention
        query = query.permute(0, 2, 1)  # [B, L, C]
        key = key.permute(0, 2, 1)  # [B, L, C]
        value = value.permute(0, 2, 1)  # [B, L, C]
        
        # Attention score
        attention = torch.bmm(query, key.transpose(1, 2))  # [B, L, L]
        attention = self.softmax(attention)
        
        # Apply attention weights
        out = torch.bmm(attention, value)  # [B, L, C]
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        return out

# ========== Define Multi-Head Self-Attention Module ==========
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x: [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)
        
        B, L, C = x.shape
        
        # [B, L, C] -> [B, L, 3*C] -> [B, L, 3, num_heads, C//num_heads]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads)
        
        # [B, L, 3, num_heads, C//num_heads] -> [3, B, num_heads, L, C//num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Separate Q, K, V 
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, C//num_heads]
        
        # Attention 
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Get output values
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)  # [B, L, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        return x

# ========== Define Multiscale Attention Feature Fusion Module (NEW) ==========
class MultiscaleAttentionFusion(nn.Module):
    """
    Multiscale Attention Feature Fusion (MAF) Module for 1D signals
    
    Args:
        channels (int): Number of input channels (C)
        reduction_ratio (int): Channel reduction ratio for the bottleneck (r)
    """
    def __init__(self, channels, reduction_ratio=16):
        super(MultiscaleAttentionFusion, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Global branch components (left side)
        self.global_gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        
        # Point-wise convolutions for global branch
        self.global_conv1 = nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_conv2 = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        self.global_sigmoid = nn.Sigmoid()
        
        # Point-wise convolutions for local branch (right side)
        self.local_conv1 = nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        self.local_sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): First input tensor of shape [B, C, L] (X in diagram)
            y (torch.Tensor): Second input tensor of shape [B, C, L] (Y in diagram)
            
        Returns:
            torch.Tensor: Fused output tensor (Z in diagram)
        """
        batch_size, C, L = x.size()
        
        # Global branch (left side) - Extract global feature information
        x_global = x  # Original path
        
        # Global attention pathway
        u = self.global_gap(x)  # [B, C, 1]
        u = self.global_conv1(u)  # [B, C/r, 1]
        u = self.global_relu(u)
        u = self.global_conv2(u)  # [B, C, 1]
        k = self.global_sigmoid(u)  # Attention weights K
        
        # Apply global attention
        x_reweighted = x_global * k  # Element-wise multiplication for re-weighting
        y_reweighted = y * k  # Element-wise multiplication for re-weighting
        
        # First global output Z1 (left branch)
        z1 = x_reweighted + y_reweighted
        
        # Local branch (right side) - Extract local detail information
        x_local = x  # Skip connection
        y_local = y  # Skip connection
        
        # Local attention pathway
        m = self.local_conv1(y)  # [B, C/r, L]
        m = self.local_relu(m)
        m = self.local_conv2(m)  # [B, C, L]
        k_star = self.local_sigmoid(m)  # Spatial attention map K*
        
        # Apply local attention
        x_local_reweighted = x_local * k_star  # Element-wise multiplication
        y_local_reweighted = y_local * k_star  # Element-wise multiplication
        
        # Second local output Z2 (right branch)
        z2 = x_local_reweighted + y_local_reweighted
        
        # Final fusion
        z = z1 + z2
        
        return z

# ========== Define Encoder Block ==========
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(EncoderBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)  # Downsample by factor of 2
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        self.mhsa = MultiHeadSelfAttention(out_channels, num_heads=num_heads)
        
    def forward(self, x):
        # Convolutional block
        conv_out = self.conv_block(x)
        
        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(conv_out)
        temp_attn = self.temporal_attention(conv_out)
        conv_attn = conv_out + channel_attn + temp_attn  # Residual connection
        
        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(conv_attn)
        refined = conv_attn + mhsa_out  # Residual connection
        
        return refined, conv_out  # Return both refined output and intermediate feature for skip connection

# ========== Define Decoder Block ==========
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, num_heads=4):
        super(DecoderBlock, self).__init__()
        
        # Upsampling conv transpose
        self.up_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        # Skip connection fusion (if applicable)
        self.has_skip = skip_channels > 0
        if self.has_skip:
            self.skip_fusion = MultiscaleAttentionFusion(out_channels)
            
        # Convolutional block after upsampling (and optional skip fusion)
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        self.mhsa = MultiHeadSelfAttention(out_channels, num_heads=num_heads)
        
    def forward(self, x, skip=None):
        # Upsampling
        up = self.up_conv(x)
        
        # Skip connection fusion
        if self.has_skip and skip is not None:
            # Ensure skip connection matches upsampled feature map size
            if skip.size(2) != up.size(2):
                skip = nn.functional.adaptive_max_pool1d(skip, up.size(2))
            
            up = self.skip_fusion(up, skip)
        
        # Convolutional block
        conv_out = self.conv_block(up)
        
        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(conv_out)
        temp_attn = self.temporal_attention(conv_out)
        conv_attn = conv_out + channel_attn + temp_attn  # Residual connection
        
        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(conv_attn)
        refined = conv_attn + mhsa_out  # Residual connection
        
        return refined

# ========== Define Bottleneck Block ==========
class BottleneckBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(BottleneckBlock, self).__init__()
        
        # Bottleneck convolutions
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels*2, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms for bottleneck
        self.channel_attention = ChannelAttention(channels)
        self.temporal_attention = TemporalAttention(channels)
        self.mhsa = MultiHeadSelfAttention(channels, num_heads=num_heads)
        
    def forward(self, x):
        # Convolutional bottleneck
        bottleneck = self.conv_block(x)
        
        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(bottleneck)
        temp_attn = self.temporal_attention(bottleneck)
        bottleneck_attn = bottleneck + channel_attn + temp_attn  # Residual connection
        
        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(bottleneck_attn)
        refined = bottleneck_attn + mhsa_out  # Residual connection
        
        return refined

# ========== Define Complete EEG Reconstruction Model ==========
class EEGReconstructionModel(nn.Module):
    def __init__(self, input_channels=1, input_length=19, base_channels=32, latent_dim=256, num_heads=4):
        super(EEGReconstructionModel, self).__init__()
        
        self.input_length = input_length
        
        # Encoder pathway
        self.encoder1 = EncoderBlock(input_channels, base_channels, num_heads=num_heads)  # Out: base_channels, length/2
        self.encoder2 = EncoderBlock(base_channels, base_channels*2, num_heads=num_heads)  # Out: base_channels*2, length/4
        self.encoder3 = EncoderBlock(base_channels*2, base_channels*4, num_heads=num_heads)  # Out: base_channels*4, length/8
        
        # Bottleneck
        self.bottleneck = BottleneckBlock(base_channels*4, num_heads=num_heads)  # Out: base_channels*4, length/8
        
        # Feature fusion for bottleneck features (now using MultiscaleAttentionFusion)
        self.fusion = MultiscaleAttentionFusion(base_channels*4)
        
        # Linear projection to latent space and back
        # Calculate flattened size based on input length
        self.bottleneck_size = base_channels*4 * (input_length // 8)
        
        # Projection to latent space and back
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_size, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.bottleneck_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder pathway
        self.decoder3 = DecoderBlock(base_channels*4, base_channels*2, skip_channels=base_channels*4, num_heads=num_heads)  # Out: base_channels*2, length/4
        self.decoder2 = DecoderBlock(base_channels*2, base_channels, skip_channels=base_channels*2, num_heads=num_heads)  # Out: base_channels, length/2
        self.decoder1 = DecoderBlock(base_channels, base_channels//2, skip_channels=base_channels, num_heads=num_heads)  # Out: base_channels//2, length
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv1d(base_channels//2, input_channels, kernel_size=1),
            nn.Tanh()  # Tanh to constrain output to [-1, 1] range, common for EEG signals
        )
        
    def encode(self, x):
        # Encoding path
        enc1, skip1 = self.encoder1(x)
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        
        # Project to latent space
        latent = self.to_latent(bottleneck)
        
        return latent, [skip1, skip2, skip3]
    
    def decode(self, latent, skip_connections):
        # Reconstruct from latent space
        bottleneck_flat = self.from_latent(latent)
        bottleneck = bottleneck_flat.view(-1, self.bottleneck_size // (self.input_length // 8), self.input_length // 8)
        
        # Apply fusion at bottleneck for additional refinement
        bottleneck = self.fusion(bottleneck, bottleneck)
        
        # Decoding path with skip connections
        dec3 = self.decoder3(bottleneck, skip_connections[2])
        dec2 = self.decoder2(dec3, skip_connections[1])
        dec1 = self.decoder1(dec2, skip_connections[0])
        
        # Output layer
        output = self.output_layer(dec1)
        
        return output
    
    def forward(self, x):
        # Encoding
        latent, skip_connections = self.encode(x)
        
        # Decoding
        output = self.decode(latent, skip_connections)
        
        return output, latent

# ========== Loss Functions for EEG Reconstruction ==========
class EEGReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        super(EEGReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for L1 loss
        self.gamma = gamma  # Weight for frequency domain loss
        
    def frequency_loss(self, x, y):
        # Convert signals to frequency domain using FFT
        x_fft = torch.fft.rfft(x, dim=2)
        y_fft = torch.fft.rfft(y, dim=2)
        
        # Compute mean squared error in frequency domain
        return torch.mean(torch.abs(x_fft - y_fft)**2)
    
    def forward(self, pred, target):
        # Time domain losses
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        # Frequency domain loss
        freq_loss = self.frequency_loss(pred, target)
        
        # Combined loss
        total_loss = self.alpha * mse + self.beta * l1 + self.gamma * freq_loss
        
        return total_loss

# ========== Initialize Model ==========
def create_eeg_reconstruction_model(input_channels=1, input_length=19, base_channels=32, latent_dim=256, num_heads=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGReconstructionModel(
        input_channels=input_channels,
        input_length=input_length,
        base_channels=base_channels,
        latent_dim=latent_dim,
        num_heads=num_heads
    ).to(device)
    
    return model, EEGReconstructionLoss().to(device)