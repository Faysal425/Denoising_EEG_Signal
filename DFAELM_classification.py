import torch
import torch.nn as nn
import torch.optim as optim

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

# ========== Define Multiscale Attention Fusion Module ==========
class MultiscaleAttentionFusion(nn.Module):
    """
    Multiscale Attention Feature Fusion (MAF) Module for 1D data
    
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
            x (torch.Tensor): First input tensor of shape [B, C, L] 
            y (torch.Tensor): Second input tensor of shape [B, C, L]
            
        Returns:
            torch.Tensor: Fused output tensor
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

# ========== Define Extreme Learning Machine Classifier ==========
class ELMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ELMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Hidden layer with random weights (not trainable)
        self.hidden_layer = nn.Linear(input_size, hidden_size, bias=True)
        
        # Initialize weights randomly and freeze them
        self._init_weights()
        self._freeze_hidden_layer()
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def _init_weights(self):
        # Initialize hidden layer weights randomly
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
    
    def _freeze_hidden_layer(self):
        # Freeze the hidden layer parameters
        for param in self.hidden_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Hidden layer feature mapping with frozen weights
        hidden_output = self.activation(self.hidden_layer(x))
        
        # Output layer
        output = self.output_layer(hidden_output)
        return output
    
    def fit(self, X, y, alpha=0.001):
        """
        Analytically compute the output layer weights using pseudoinverse
        
        Args:
            X: Input features (tensor)
            y: Target labels (tensor)
            alpha: Regularization parameter
        """
        # Get hidden layer output
        with torch.no_grad():
            hidden_output = self.activation(self.hidden_layer(X))
        
        # Convert target to one-hot encoded format if needed
        if y.dim() == 1:
            y_onehot = torch.zeros(y.size(0), self.num_classes, device=y.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
        else:
            y_onehot = y
        
        # Calculate output weights using ridge regression (pseudoinverse)
        H = hidden_output
        H_T = H.t()
        
        # Ridge regression solution with regularization
        # W = (H^T * H + alpha * I)^(-1) * H^T * T
        identity = torch.eye(H.size(1), device=H.device) * alpha
        temp = torch.inverse(H_T @ H + identity) @ H_T
        output_weights = temp @ y_onehot
        
        # Set output layer weights
        with torch.no_grad():
            self.output_layer.weight.copy_(output_weights.t())
            # Bias will be learned through backpropagation

# ========== Define Enhanced CNN Model with Attention, Residual Connections, MAF and ELM ==========
class EnhancedHeartNetIEEE(nn.Module):
    def __init__(self, input_size=28, num_classes=2, num_heads=4, elm_hidden_size=512):
        super(EnhancedHeartNetIEEE, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1)
        )
        
        # Channel attention and temporal attention after first block
        self.channel_attention1 = ChannelAttention(64)
        self.temporal_attention1 = TemporalAttention(64)
        
        # Multi-head self-attention for feature refinement
        self.mhsa1 = MultiHeadSelfAttention(64, num_heads=num_heads)
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1)
        )
        
        # Channel attention and temporal attention after second block
        self.channel_attention2 = ChannelAttention(128)
        self.temporal_attention2 = TemporalAttention(128)
        
        # Multi-head self-attention for feature refinement
        self.mhsa2 = MultiHeadSelfAttention(128, num_heads=num_heads)
        
        # Define expansion convolutions as module components
        self.conv1_expander = nn.Conv1d(64, 256, kernel_size=1)
        self.conv2_expander = nn.Conv1d(128, 256, kernel_size=1)
        
        # Feature fusion module - Multiscale Attention Fusion
        self.feature_fusion = MultiscaleAttentionFusion(256, reduction_ratio=16)
        
        # Calculate output size after convolutions for the fully connected layer
        self.flatten_size = self._get_flatten_size(input_size)
        
        # Feature extractor (replaced classifier)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # ELM classifier
        self.elm_classifier = ELMClassifier(256, elm_hidden_size, num_classes)
        
        # Flag to track if ELM has been fitted
        self.elm_fitted = False

    def _get_flatten_size(self, input_size):
        # Calculate the output size after all convolution operations
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            
            # First convolution block
            out1 = self.conv_block1(dummy_input)
            
            # Apply channel and temporal attention with residual connections
            channel_attn1 = self.channel_attention1(out1)
            temp_attn1 = self.temporal_attention1(out1)
            out1_attn = out1 + channel_attn1 + temp_attn1  # Residual connection
            
            # Apply multi-head attention with residual connection
            mhsa_out1 = self.mhsa1(out1_attn)
            out1_refined = out1_attn + mhsa_out1  # Residual connection
            
            # Second convolution block
            out2 = self.conv_block2(out1_refined)
            
            # Apply channel and temporal attention with residual connections
            channel_attn2 = self.channel_attention2(out2)
            temp_attn2 = self.temporal_attention2(out2)
            out2_attn = out2 + channel_attn2 + temp_attn2  # Residual connection
            
            # Apply multi-head attention with residual connection
            mhsa_out2 = self.mhsa2(out2_attn)
            out2_refined = out2_attn + mhsa_out2  # Residual connection
            
            print(f"Shape after conv1 with residuals: {out1_refined.shape}")
            print(f"Shape after conv2 with residuals: {out2_refined.shape}")
            
            # Apply expansion
            out1_expanded = self.conv1_expander(out1_refined)
            out2_expanded = self.conv2_expander(out2_refined)
            
            # Check if there's a size mismatch and adjust if needed
            if out1_expanded.shape[2] != out2_expanded.shape[2]:
                # Use the smaller dimension as target
                target_size = min(out1_expanded.shape[2], out2_expanded.shape[2])
                out1_expanded = nn.functional.adaptive_max_pool1d(out1_expanded, target_size)
                out2_expanded = nn.functional.adaptive_max_pool1d(out2_expanded, target_size)
            
            # Feature fusion with MultiscaleAttentionFusion
            fused = self.feature_fusion(out1_expanded, out2_expanded)
            print(f"Shape after fusion: {fused.shape}")
            
            return fused.shape[1] * fused.shape[2]

    def forward(self, x):
        # First convolutional block
        conv1_out = self.conv_block1(x)
        
        # Apply channel and temporal attention with residual connections
        channel_attn1 = self.channel_attention1(conv1_out)
        temp_attn1 = self.temporal_attention1(conv1_out)
        conv1_attn = conv1_out + channel_attn1 + temp_attn1  # Residual connection
        
        # Apply multi-head attention with residual connection
        mhsa_out1 = self.mhsa1(conv1_attn)
        conv1_refined = conv1_attn + mhsa_out1  # Residual connection
        
        # Second convolutional block
        conv2_out = self.conv_block2(conv1_refined)
        
        # Apply channel and temporal attention with residual connections
        channel_attn2 = self.channel_attention2(conv2_out)
        temp_attn2 = self.temporal_attention2(conv2_out)
        conv2_attn = conv2_out + channel_attn2 + temp_attn2  # Residual connection
        
        # Apply multi-head attention with residual connection
        mhsa_out2 = self.mhsa2(conv2_attn)
        conv2_refined = conv2_attn + mhsa_out2  # Residual connection
        
        # Create 256-channel tensors using the defined expanders
        conv1_expanded = self.conv1_expander(conv1_refined)
        conv2_expanded = self.conv2_expander(conv2_refined)
        
        # Check if there's a size mismatch and adjust if needed
        if conv1_expanded.shape[2] != conv2_expanded.shape[2]:
            # Use the smaller dimension as target
            target_size = min(conv1_expanded.shape[2], conv2_expanded.shape[2])
            conv1_expanded = nn.functional.adaptive_max_pool1d(conv1_expanded, target_size)
            conv2_expanded = nn.functional.adaptive_max_pool1d(conv2_expanded, target_size)
        
        # Feature fusion using MultiscaleAttentionFusion
        fused = self.feature_fusion(conv1_expanded, conv2_expanded)
        
        # Flatten 
        flat = fused.view(fused.size(0), -1)
        
        # Extract features
        features = self.feature_extractor(flat)
        
        # ELM classification
        output = self.elm_classifier(features)
        
        return output
    
    def fit_elm(self, dataloader, device):
        """
        Fit the ELM classifier using the extracted features from the trained CNN
        """
        self.eval()  # Set model to evaluation mode
        all_features = []
        all_labels = []
        
        # Extract features and labels from dataloader
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Get features up to the feature extraction layer
                conv1_out = self.conv_block1(inputs)
                channel_attn1 = self.channel_attention1(conv1_out)
                temp_attn1 = self.temporal_attention1(conv1_out)
                conv1_attn = conv1_out + channel_attn1 + temp_attn1
                mhsa_out1 = self.mhsa1(conv1_attn)
                conv1_refined = conv1_attn + mhsa_out1
                
                conv2_out = self.conv_block2(conv1_refined)
                channel_attn2 = self.channel_attention2(conv2_out)
                temp_attn2 = self.temporal_attention2(conv2_out)
                conv2_attn = conv2_out + channel_attn2 + temp_attn2
                mhsa_out2 = self.mhsa2(conv2_attn)
                conv2_refined = conv2_attn + mhsa_out2
                
                # Create 256-channel tensors using the defined expanders
                conv1_expanded = self.conv1_expander(conv1_refined)
                conv2_expanded = self.conv2_expander(conv2_refined)
                
                # Check if there's a size mismatch and adjust if needed
                if conv1_expanded.shape[2] != conv2_expanded.shape[2]:
                    # Use the smaller dimension as target
                    target_size = min(conv1_expanded.shape[2], conv2_expanded.shape[2])
                    conv1_expanded = nn.functional.adaptive_max_pool1d(conv1_expanded, target_size)
                    conv2_expanded = nn.functional.adaptive_max_pool1d(conv2_expanded, target_size)
                
                # Feature fusion using MultiscaleAttentionFusion
                fused = self.feature_fusion(conv1_expanded, conv2_expanded)
                flat = fused.view(fused.size(0), -1)
                features = self.feature_extractor(flat)
                
                all_features.append(features)
                all_labels.append(labels)
        
        # Concatenate all features and labels
        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)
        
        # Fit ELM classifier
        self.elm_classifier.fit(all_features, all_labels, alpha=0.01)
        self.elm_fitted = True
        
        print("ELM classifier fitted successfully!")