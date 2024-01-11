import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


#  Extract the crucial last hidden state 
def last_hidden_states(backbone_outs):
    """
    Retrieves the final hidden state from the backbone's output,
    containing the most refined contextual information.
    """

    last_hidden_state = backbone_outs[0]  # ➡️➡️➡️ Access the last hidden state directly
    return last_hidden_state  #  Return it for further exploration


#  Unearth all hidden states for thorough analysis 
def all_hidden_states(backbone_outs):
    """
    Gathers all hidden states generated by the backbone,
    providing a comprehensive view of the model's internal processing.
    """

    all_hidden_states = torch.stack(backbone_outs[1])  # ️️️ Stack them neatly together
    return all_hidden_states  #  Offer them for deep insights


#  Retrieve the input IDs for text representation 
def input_ids(inputs):
    """
    Retrieves the input IDs representing the text input,
    essential for language processing tasks.
    """

    return inputs['input_ids']  #  Unlock the numerical text encoding


#  Obtain the attention mask for focused comprehension 
def get_attention_mask(inputs):
    """
    Retrieves the attention mask, crucial for guiding the model's focus
    towards relevant parts of the input sequence.
    """

    return inputs['attention_mask']  #  Shine a light on important words


# ✨ MeanPooling Module ✨
class MeanPooling(nn.Module):
    """
    https://arxiv.org/pdf/1811.00202.pdf

    Calculates mean embeddings for text sequences, carefully considering masked elements.

     Key attributes:
    - output_dim: The dimensionality of the output embeddings.
    """

    def __init__(self, backbone_config, pooling_config):
        super().__init__()  # Inherit from parent class
        self.output_dim = backbone_config.hidden_size  # Set output dimension

    #  Forward pass 
    def forward(self, inputs, backbone_outs):
        """
        Performs the mean pooling operation.

        Args:
            inputs: The input sequence data.
            backbone_outs: The outputs from the backbone model.

        Returns:
            torch.Tensor: The calculated mean embeddings.
        """

        #  Extract attention mask 
        attention_mask = get_attention_mask(inputs)  # Obtain attention mask

        # ️ Access last hidden state ️
        last_hidden_state = last_hidden_states(backbone_outs)  # Retrieve last hidden state

        # ✨ Mask expansion for precise weighting ✨
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # Expand mask for element-wise multiplication

        # ⚖️ Weighted sum calculation ⚖️
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)  # Calculate weighted sum

        #  Mask summation for normalization 
        sum_mask = input_mask_expanded.sum(1)  # Sum mask elements
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Ensure non-zero denominator

        # ➗ Calculate mean embeddings ➗
        mean_embeddings = sum_embeddings / sum_mask  # Divide by mask sum for mean calculation

        return mean_embeddings  # Return the calculated mean embeddings


# ✨ LSTMPooling: Sequence Synthesizer ✨
class LSTMPooling(nn.Module):
    """
    Harnesses the power of Long Short-Term Memory (LSTM) to meticulously extract and condense crucial
    information from a sequence of hidden states, crafting a refined representation for downstream tasks.
    """

    def __init__(self, backbone_config, pooling_config, is_lstm=True):
        super().__init__()  # Inherit superpowers from parent ‍♀️

        # ️ Extract configuration
        self.hidden_layer = backbone_config.hidden_layer  # Number of hidden layers in backbone
        self.hidden_size = backbone_config.hidden_size  # Hidden size of backbone layers
        self.hidden_lstm_size = pooling_config.hidden_size  # Hidden size for LSTM/GRU
        self.dropout_rate = pooling_config.dropout_rate  # Regularization for generalization 
        self.bidirectional = pooling_config.bidirectional  # Enable bidirectional exploration 

        # ⚙️ Core LSTM/GRU configuration ⚙️
        self.is_lstm = is_lstm  # Choose LSTM or GRU for sequence magic ✨
        self.output_dim = pooling_config.hidden_size * 2 if self.bidirectional else pooling_config.hidden_size  # Output dimension

        #  Construct the chosen sequence wizard 
        if self.is_lstm:
            self.lstm = nn.LSTM(
                self.hidden_size,  # Input dimension
                self.hidden_lstm_size,  # Hidden dimension
                bidirectional=self.bidirectional,  # Enable bidirectionality
                batch_first=True  # Prioritize batch processing ️
            )
        else:
            self.lstm = nn.GRU(
                self.hidden_size,  # Input dimension
                self.hidden_lstm_size,  # Hidden dimension
                bidirectional=self.bidirectional,  # Enable bidirectionality
                batch_first=True  # Prioritize batch processing ️
            )

        #  Occasional memory wipes for enhanced focus and generalization
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outs):
        """
        Orchestrates the intricate of information extraction and refinement:

        1. Gathers hidden states from the backbone layers ️‍♀️
        2. Meticulously arranges them for LSTM/GRU processing  ️
        3. Unleashes the sequence  to capture long-range dependencies 
        4. Applies a final memory-sharpening dropout 
        5. Presents the distilled knowledge for further insights 
        """

        all_hidden_states = all_hidden_states(backbone_outs)  # Gather hidden states ️

        hidden_states = torch.stack([
            all_hidden_states[layer_i][:, 0].squeeze()  # Focus on first token 
            for layer_i in range(1, self.hidden_layer + 1)  # Iterate through hidden layers
        ], dim=-1)  # Arrange for LSTM/GRU ️

        hidden_states = hidden_states.view(-1, self.hidden_layer, self.hidden_size)  # Reshape for processing 

        # ✨ LSTM/GRU weaves its magic ✨
        out, _ = self.lstm(hidden_states, None)  # Unleash the sequence wizard ‍♀️

        out = self.dropout(out[:, -1, :])  # Apply final memory sharpening 

        return out  # Present the refined knowledge 


# ✨✨✨ Weighted Layer Pooling for Precision Blending ✨✨✨
class WeightedLayerPooling(nn.Module):
    """
    Crafts a refined representation by meticulously combining hidden states from multiple layers,
    applying carefully calculated weights to each layer for optimal balance. ⚖️
    """

    def __init__(self, backbone_config, pooling_config):
        super().__init__()  # Inherit powers from the parent module 

        #  Grasp essential configuration details 
        self.hidden_layer = backbone_config.hidden_layer  # Number of concealed layers ️
        self.start = pooling_config.start  # Designated starting point for extraction 🪜
        
        # ⚖️ Prepare layer weights for precise blending ⚖️
        self.layer_weights = pooling_config.layer_weights if pooling_config.layer_weights is not None else \
            nn.Parameter(torch.tensor([1] * (self.hidden_layer + 1 - self.start), dtype=torch.float))  # Initialize weights if not provided ⚖️

        self.output_dim = backbone_config.hidden_size  # Dimension of the refined output 

    def forward(self, inputs, backbone_outs):
        """
        Orchestrates the meticulous blending process ✨

        Args:
            inputs (torch.Tensor): Input data to be processed.
            backbone_outs (list): Hidden states from the backbone model.

        Returns:
            torch.Tensor: A refined representation forged through weighted layer pooling.
        """

        # 🪔 Unveil all hidden states 🪔
        all_hidden_states = all_hidden_states(backbone_outs)  # Gather hidden knowledge from each layer 

        # ✂️ Extract relevant layers for precise blending ✂️
        all_layer_embedding = all_hidden_states[self.start:, :, :, :]  # Slice the chosen layers for the elixir ⚗️

        # ⚖️ Apply meticulous weights to each layer ⚖️
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())  # Stretch weights to match layer dimensions 
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()  # Calculate the weighted average ⚖️

        #  Return the refined essence 
        return weighted_average[:, 0]  # Deliver the concentrated knowledge ✨


class ConcatPooling(nn.Module):
    """
    Meticulously combines multiple hidden states to create a unified representation ️️️
    """

    def __init__(self, backbone_config, pooling_config):
        super().__init__()  # Inheriting the best from parent classes 

        self.n_layers = pooling_config.n_layers  # Number of layers to incorporate 
        self.output_dim = backbone_config.hidden_size * pooling_config.n_layers  # Calculating the final output dimension 

    def forward(self, inputs, backbone_outs):
        """
         Dynamically performs concatenation for enhanced representation 
        """

        all_hidden_states = all_hidden_states(backbone_outs)  # Gather all the hidden states 

        concatenate_pooling = torch.cat([
            all_hidden_states[-(i + 1)] for i in range(self.n_layers)  # Concatenation for a grander perspective 
        ], -1)  # Combining along the specified dimension 

        concatenate_pooling = concatenate_pooling[:, 0]  # Selecting a specific portion for focus 
        return concatenate_pooling  # Returning the carefully crafted result 

# ✨ AttentionPooling: Prioritizing Information for Enhanced Insights ✨
class AttentionPooling(nn.Module):
    """
    Implements a sophisticated attention pooling mechanism to selectively focus on the 
    most informative elements within input sequences, enhancing model performance.
    """

    def __init__(self, backbone_config, pooling_config):
        super().__init__()

        #  Access model configuration parameters
        self.hidden_layer = backbone_config.hidden_layer
        self.hidden_size = backbone_config.hidden_size
        self.hidden_dim_fc = pooling_config.hidden_dim_fc
        self.dropout = nn.Dropout(pooling_config.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #  Device optimization

        #  Initialize learnable parameters
        self.q = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size)))).float().to(self.device)
        self.w_h = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hidden_dim_fc)))).float().to(self.device)

        self.output_dim = self.hidden_dim_fc

    # ➡️ Forward Pass: Orchestrating Information Flow ➡️
    def forward(self, inputs, backbone_outs):
        """
        Orchestrates the attention pooling process, producing a refined representation.
        """

        # ️ Retrieve hidden states from backbone model
        all_hidden_states = all_hidden_states(backbone_outs)

        #  Strategically select hidden states
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze() 
                                    for layer_i in range(1, self.hidden_layer + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.hidden_layer, self.hidden_size)

        # ✨✨✨ Spotlight on attention ✨✨✨
        out = self.attention(hidden_states)
        out = self.dropout(out)  # ️ Regularization for robustness
        return out
    
    #  Core Attention Mechanism 
    def attention(self, h):
        """
        Meticulously calculates attention weights and produces a weighted representation.
        """

        #  Calculate attention scores
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = nn.functional.softmax(v, -1)  # ⚖️ Normalize scores

        # ⚖️ Weigh hidden states based on attention
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class SBERTWKPooling(nn.Module):
    """
     This class holds the secrets to expertly pooling information from hidden states 

    Pooling based on the paper: "SBERT-WK: A Sentence Embedding Method ByDissecting BERT-based Word Models"
    https://arxiv.org/pdf/2002.06652.pdf

    Note: SBERT-WK uses QR decomposition. torch QR decomposition is currently extremely slow when run on GPU.
    Hence, the tensor is first transferred to the CPU before it is applied. This makes this pooling method rather slow
    """

    def __init__(self, backbone_config, pooling_config):
        super().__init__()  # ✨✨✨ Summoning the powers of the parent class ✨✨✨

        # ‍♀️ Parameters for pooling wizardry ‍♀️
        self.start = pooling_config.start
        self.context_window_size = pooling_config.context_window_size
        self.output_dim = backbone_config.hidden_size  #  Output dimension for magical transformations 

    def forward(self, inputs, backbone_outs):
        """
         Unleashing the pooling power! 
        """

        all_hidden_states = all_hidden_states(backbone_outs)  #  Gathering hidden states from the model's depths 
        attention_mask = get_attention_mask(inputs)  #  Focusing on the truly important elements 

        #  Reshaping and shifting for optimal pooling 
        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device  #  Remembering the original device for later 
        all_layer_embedding = ft_all_layers.transpose(1, 0)  # ‍♀️‍♀️‍♀️ A little flip for fun and flexibility ‍♀️‍♀️‍♀️
        all_layer_embedding = all_layer_embedding[:, self.start:, :, :]  # ✂️✂️✂️ Trimming for precision ✂️✂️✂️

        # ✈️✈️✈️ Taking a brief trip to the CPU for efficient pooling ✈️✈️✈️
        all_layer_embedding = all_layer_embedding.cpu()

        #  Aligning with attention for deeper insights 
        attention_mask = attention_mask.cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1

        #  Crafting sentence embeddings with care 
        embedding = []
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []

            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                token_embedding = self.unify_token(token_feature)  # ✨✨✨ Uniting token knowledge ✨✨✨
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)  #  Merging sentence wisdom 
            embedding.append(sentence_embedding)

        #  Returning to the original device for further adventures 
        output_vector = torch.stack(embedding).to(org_device)
        return output_vector  #  Presenting the pooled knowledge with a flourish! 


# ✨ MeanMaxPooling Class: The Master of Combining Averages and Extremes ✨
class MeanMaxPooling(nn.Module):
    """
    This class takes a sequence of embeddings and strategically extracts both their mean  and maximum  values,
    creating a powerful representation that captures both central tendencies and key highlights. 
    """

    def __init__(self, backbone_config, pooling_config):  # Initialization for smooth sailing ⛵
        super().__init__()  # Inheriting superpowers from the parent class 
        self.feat_mult = 1  # Setting a multiplier for features, ready for action ⚡
        self.output_dim = backbone_config.hidden_size  # Defining the dimension for output 

    def forward(self, inputs, backbone_outs):  # The heart of the magic 
        # 1️⃣ Gather essential ingredients ‍♂️
        attention_mask = get_attention_mask(inputs)  # Retrieve the attention mask 
        x = input_ids(inputs)  # Grab the input IDs for embedding action 

        # 2️⃣ Craft the attention mask for calculations ️
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()  # Expand it for flexibility ‍♀️

        # 3️⃣ Calculate the mean embeddings ⚖️
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)  # Sum up with mask for precision 
        sum_mask = input_mask_expanded.sum(1)  # Total up the mask values for normalization ‍♀️
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Gently safeguard against zero division ➗
        mean_embeddings = sum_embeddings / sum_mask  # Unveil the average embeddings ✨

        # 4️⃣ Find the max embeddings 
        embeddings = x.clone()  # Create a copy for extreme extraction 
        embeddings[input_mask_expanded == 0] = -1e4  # Mask out irrelevant elements 
        max_embeddings, _ = torch.max(embeddings, dim=1)  # Unleash the maximum values 

        # 5️⃣ Unite mean and max embeddings for a dynamic duo 
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)  # Combine their strengths 

        return mean_max_embeddings  # Return the enriched representation for further adventures 

#  MaxPooling Class: Ready to Find the Maximum! 
class MaxPooling(nn.Module):
    """
    This class is designed to expertly extract the most significant features ✨ from input sequences
    by applying the powerful technique of maximum pooling .
    """

    def __init__(self, backbone_config, pooling_config):
        super(MaxPooling, self).__init__()  # Inheriting essential superpowers ‍♀️

        #  Key Parameters 
        self.feat_mult = 1  # A multiplier for feature dimensions 
        self.output_dim = backbone_config.hidden_size  # The dimension of output embeddings 

    def forward(self, inputs, backbone_outs):
        """
        This method orchestrates the pooling process , carefully extracting maximum values .

        Args:
            inputs (torch.Tensor): The input sequences, brimming with information 
            backbone_outs (torch.Tensor): Additional outputs from the backbone model 

        Returns:
            torch.Tensor: The distilled embeddings, capturing the essence of the inputs 
        """

        #  Obtaining Attention 
        attention_mask = get_attention_mask(inputs)  # Focusing on the relevant parts 

        #  Extracting Input IDs 
        x = input_ids(inputs)  # Accessing the fundamental input IDs ️

        #  Expanding Attention 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()  # Widening the attention lens 

        #  Cloning for Protection 
        embeddings = x.clone()  # Safeguarding the original embeddings ️

        #  Silencing the Unimportant 
        embeddings[input_mask_expanded == 0] = -1e4  # Muting those not worthy of attention 

        #  Crowning the Maximum 
        max_embeddings, _ = torch.max(embeddings, dim=1)  # Identifying and extracting the most significant values 

        return max_embeddings  # Returning the triumphant results 


#   MinPooling: Capturing the Essence of Input Sequences 
class MinPooling(nn.Module):
    """
    Captures the most essential elements within input sequences by masterfully employing minimum pooling operations.
    """

    def __init__(self, backbone_config, pooling_config):
        """
        Initializes the MinPooling module with key configurations.

        Args:
            backbone_config (dict): Configuration details for the model's backbone.
            pooling_config (dict): Configuration settings specific to the pooling process.
        """

        super(MinPooling, self).__init__()  # Invoke the parent class's constructor
        self.feat_mult = 1  # Feature multiplier for dimensional adjustments
        self.output_dim = backbone_config.hidden_size  # Output dimension for consistency

    def forward(self, inputs, backbone_outs):
        """
        Performs the core forward pass through the MinPooling module.

        Args:
            inputs (tensor): Input sequences to be processed.
            backbone_outs (tensor): Output generated from the model's backbone.

        Returns:
            tensor: The pooled embeddings, meticulously crafted to represent the most significant aspects of the input.
        """

        #  Create an attention mask to focus on relevant elements 
        attention_mask = get_attention_mask(inputs)  # Obtain the attention mask

        #  Extract input IDs for further processing 
        x = input_ids(inputs)

        #  Expand the attention mask for precise embedding manipulation 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()

        #  Create a copy of the embeddings for safe experimentation 
        embeddings = x.clone()

        # 🪄 Strategically mask irrelevant elements with a tiny value 🪄
        embeddings[input_mask_expanded == 0] = 1e-4  # Silence those that need not be heard

        # ✨✨✨ Unleash the power of minimum pooling! ✨✨✨
        min_embeddings, _ = torch.min(embeddings, dim=1)  # Capture the most essential features

        return min_embeddings  # Return the distilled essence of the input sequences, ready for further analysis


#  GeMText: Master of Textual Pooling 
class GeMText(nn.Module):
    """
    Harnesses the power of Generalized Mean (GeM) pooling for effective extraction of key information
    from textual data. ✨
    """

    def __init__(self, backbone_config, pooling_config):
        super(GeMText, self).__init__()  # Inherit essential features from parent class ✨

        #  Key parameters for pooling precision 
        self.dim = pooling_config.dim  # Target dimension for pooling
        self.eps = pooling_config.eps  # Small value for numerical stability 
        self.feat_mult = 1  # Feature multiplier ⚙️

        # ️ Learnable parameter for adaptive pooling ️
        self.p = Parameter(torch.ones(1) * pooling_config.p)

        # ✨ Output dimension aligned with backbone ✨
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_output):
        """
        Performs the magical GeM pooling operation on textual data. 
        """

        #  Obtain attention mask for focused pooling 
        attention_mask = get_attention_mask(inputs)
        x = input_ids(inputs)  # Secure input IDs 

        #  Amplify attention mask for seamless integration 
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)

        #  Core GeM pooling operation 
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)  # Empowered pooling with attention 
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)  # Normalize for balanced representation ⚖️
        ret = ret.pow(1 / self.p)  # Final GeM transformation ✨

        return ret  # Return the pooled textual essence 


def get_pooling_layer(config, backbone_config):
    """
     Finds the perfect pooling layer for your model's needs! 

    Args:
        config (dict): The configuration dictionary containing model specifications.
        backbone_config (dict): The configuration for the backbone network.

    Returns:
        The appropriate pooling layer class, ready to be instantiated! 

    Raises:
        ValueError: If an invalid pooling type is specified. 
    """

    # ️️️ Navigate through different pooling options ️️️
    if config.model.pooling_type == 'MeanPooling':
        return MeanPooling(backbone_config, config.model.gru_pooling)  # ⚖️ Average it out for balance ⚖️

    elif config.model.pooling_type == 'GRUPooling':
        return LSTMPooling(backbone_config, config.model.gru_pooling, is_lstm=False)  # ♻️ Recurrent connections for memory ♻️

    elif config.model.pooling_type == 'LSTMPooling':
        return LSTMPooling(backbone_config, config.model.lstm_pooling, is_lstm=True)  #  Long-term memory for deeper insights 

    elif config.model.pooling_type == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_config, config.model.weighted_pooling)  # ⚖️ Assigning importance strategically ⚖️

    elif config.model.pooling_type == 'SBERTWKPooling':
        return SBERTWKPooling(backbone_config, config.model.wk_pooling)  #  Knowledge-driven pooling for focused attention 

    elif config.model.pooling_type == 'ConcatPooling':
        return ConcatPooling(backbone_config, config.model.concat_pooling)  # ➕ Merging information for a comprehensive view ➕

    elif config.model.pooling_type == 'AttentionPooling':
        return AttentionPooling(backbone_config, config.model.attention_pooling)  #  Spotlighting the most relevant features 

    else:
        raise ValueError(f'⚠️ Invalid pooling type detected! ⚠️ Please check your configuration: {config.model.pooling_type}')
