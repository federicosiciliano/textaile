import torch
import torchvision
import itertools as it

# Concept-based AutoEncoder
class TextAIle(torch.nn.Module):
    def __init__(self, encoder, decoder, concept_size, embedding_size, encoder_features, decoder_features, disentangle = False, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.concept_size = concept_size
        self.embedding_size = embedding_size

        # # Encoder MLP
        # self.encoder_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(encoder_features, encoder_features*2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(encoder_features*2, concept_size + embedding_size),
        # )

        # # Decoder MLP
        # self.decoder_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(concept_size + embedding_size, decoder_features*2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(decoder_features*2, decoder_features)
        # )

        #self.concept_linear = torch.nn.Linear(encoder.out_features, embedding_size)
        #self.concept_func = lambda x: torch.cat([torch.nn.Sigmoid(x[:,:-6]), 1*(torch.nn.Sigmoid(x[:,-6:])>0.5)])
        self.recon_func = torch.nn.Sigmoid()

        # image --> encoder --> emb_encoder --> MLP --> concepts / emb2 --> MLP --> emb_decoder --> decoder --> image

        # if vae:
        #     self.linear_mu = torch.nn.Linear(embedding_size, embedding_size)
        #     self.linear_sigma = torch.nn.Linear(embedding_size, embedding_size)
        #     self.distribution = torch.normal
        #     self.vae_encode = lambda x: self.vae(x)
        # else:
        #     self.vae_encode = lambda x: {"embedding": x}

        if disentangle:
            self.disentangle_decode_encode = lambda x,y: self.disentangle(x,y)
        else:
            self.disentangle_decode_encode = lambda x,y: {}

        # parametri dropout concepts

    def forward(self, x):
        out = self.encode(x)

        # Reconstruct input
        out.update(self.decode(torch.cat([out["concepts"], out["embedding"]], dim=-1)))

        #Encode reconstructed input
        other_out = self.encode(out["reconstructed_img"])
        out["recon_concepts"] = other_out["concepts"]
        out["recon_embedding"] = other_out["embedding"]

        # Disentangle
        out.update(self.disentangle_decode_encode(out["concepts"], out["embedding"]))

        return out
    
    def encode(self, x):
        enc = self.encoder(x) #self.encoder_mlp(self.encoder(x))
        concepts = enc[:, :-self.embedding_size] #self.concept_func(enc[:, :-self.embedding_size])
        embedding = enc[:, -self.embedding_size:]
        #out = self.vae_encode(embedding)
        return {"concepts": concepts, "embedding": embedding}

    def decode(self, x):
        return {"reconstructed_img": self.recon_func(self.decoder(x))} #self.decoder_mlp(
    
    # def vae(self, x):
    #     mu = self.linear_mu(x)
    #     sigma = torch.exp(0.5*self.linear_sigma(x))+1e-8
    #     z = self.distribution(mu, sigma)
    #     return {"embedding": z, "mu": mu, "sigma": sigma}
    
    def disentangle(self, concepts, embedding):
        # other_concepts as permutation of concepts
        other_concepts = concepts[torch.randperm(concepts.shape[0])]
        new_complete_embedding = torch.cat([other_concepts, embedding], dim=-1)
        new_recon_img = self.decode(new_complete_embedding)["reconstructed_img"]

        out = self.encode(new_recon_img)
        new_out = {"other_concepts": torch.sigmoid(other_concepts), "other_embedding": out["embedding"], "new_concepts": out["concepts"]}

        return new_out

    # def concept_func(self, x):
    #     hues = torch.sigmoid(x[:,[0,3,6,9,12]])
    #     #hues = torch.maximum(torch.minimum(hues, torch.ones_like(hues)), torch.zeros_like(hues))

    #     saturations = torch.sigmoid(x[:,[1,4,7,10,13]])

    #     values = torch.sigmoid(x[:,[2,5,8,11,14]])
        
    #     hsv = torch.cat([hues, saturations, values], dim=-1)[:,[0,5,10,1,6,11,2,7,12,3,8,13,4,9,14]]

    #     harmonies = x[:,-6:] #1*(torch.sigmoid(x[:,-6:])>0.5)

    #     return torch.cat([hsv, harmonies], dim=-1)


# Concept-based AutoEncoder
class TextAIleBottleneck(torch.nn.Module):
    def __init__(self, encoder, decoder, concept_size, embedding_size, num_colors = 216, palette_size = 5, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.concept_size = concept_size
        self.embedding_size = embedding_size
        self.palette_size = palette_size
        self.num_colors = num_colors

        concept_embs = []
        for i in range(concept_size):
            if i < palette_size:
                concept_embs.append(torch.nn.Embedding(num_colors, embedding_size))
            else:
                concept_embs.append(torch.nn.Embedding(2, embedding_size))
        self.concept_embs = torch.nn.ModuleList(concept_embs)

        self.recon_func = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.encode(x)

        # Reconstruct input
        out.update(self.decode(torch.cat([out["concepts"], out["embedding"]], dim=-1)))

        return out
    
    def encode(self, x):
        enc = self.encoder(x) #self.encoder_mlp(self.encoder(x))
        concepts = enc[:, :-self.embedding_size]

        colors = concepts[:,:int(self.palette_size*self.num_colors)].reshape(-1, self.palette_size, self.num_colors)
        colors = torch.softmax(colors, dim=-1)


        # Multiply embeddings by color weights
        concept_embeddings = []
        for color_id in range(self.palette_size):
            concept_embeddings.append((self.concept_embs[color_id].weight * colors[:,color_id][:,None]).sum())

        binary_concepts = torch.sigmoid(concepts[int(self.palette_size*self.num_colors):])
        for i in range(self.concept_size):
            if i > self.palette_size:
                print(torch.cat([binary_concepts[:,i][:,None],1-binary_concepts[:,i][:,None]], dim=-1).shape)
                concept_embeddings.append(self.concept_embs[i].weight * torch.cat([binary_concepts[:,i][:,None],1-binary_concepts[:,i][:,None]], dim=-1))

        embedding = enc[:, -self.embedding_size:]
        return {"concepts": concepts, "embedding": embedding, "concept_embeddings": concept_embeddings}

    def decode(self, x):
        return {"reconstructed_img": self.recon_func(self.decoder(x))}


class Orthogonality_Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, concepts_embeddings, embedding, weight):
        # cosine similarity loss betweem concept embeddings and embedding
        loss = 0
        for concept_embedding in concepts_embeddings:
            loss += torch.nn.functional.cosine_similarity(embedding, concept_embedding, dim=-1)
        return loss.mean()

# Concept-based AutoEncoder
class TextAIleUNet(torch.nn.Module):
    def __init__(self, n_channels, concept_size, bilinear=True, pool_size=(1, 1), disentangle=False, **kwargs):
        super().__init__()

        self.concept_size = concept_size

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(pool_size)
        self.flatten = torch.nn.Flatten()

        encoder_features = 1024 // factor * pool_size[0] * pool_size[1]
        decoder_features = 1024 // factor * pool_size[0] * pool_size[1]

        # Encoder MLP
        self.encoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(encoder_features, encoder_features*2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_features*2, concept_size),
        )

        # Decoder MLP
        self.decoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(concept_size, decoder_features*2),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_features*2, decoder_features)
        )

        self.unflatten = torch.nn.Unflatten(1, (1024 // factor, pool_size[0], pool_size[1]))
        #self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_channels))

        self.recon_func = torch.nn.Sigmoid()

        self.concept_func = lambda x: torch.sigmoid(x)

        if disentangle:
            self.disentangle_decode_encode = lambda x,*args: self.disentangle(x,*args)
        else:
            self.disentangle_decode_encode = lambda *args: {}

    def forward(self, x):
        out, (x4, x3, x2, x1) = self.encode(x)

        out.update(self.decode(out["concepts"], x4, x3, x2, x1))

        out.update(self.disentangle_decode_encode(out["concepts"], x4, x3, x2, x1))

        return out
    
    def encode(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1 = x1.mean(1,keepdim=True).repeat(1, x1.shape[1], 1, 1)
        x2 = x2.mean(1,keepdim=True).repeat(1, x2.shape[1], 1, 1)
        x3 = x3.mean(1,keepdim=True).repeat(1, x3.shape[1], 1, 1)
        x4 = x4.mean(1,keepdim=True).repeat(1, x4.shape[1], 1, 1)

        flat_emb = self.flatten(self.avg_pool(x5))

        concepts = self.concept_func(self.encoder_mlp(flat_emb))

        return {"concepts": concepts}, (x4, x3, x2, x1)
    
    def decode(self, concepts, x4, x3, x2, x1):
        dec = self.unflatten(self.decoder_mlp(concepts))
        x = self.up1(dec, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return {"reconstructed_img": self.recon_func(logits)}
    
    def disentangle(self, concepts, *args):
        other_concepts = concepts[torch.randperm(concepts.shape[0])]
        new_recon_img = self.decode(other_concepts, *args)["reconstructed_img"]

        out, _ = self.encode(new_recon_img)
        new_out = {"other_concepts": other_concepts, "new_concepts": out["concepts"]}

        return new_out
    
# Concept-based AutoEncoder
class CustomTextAIle(torch.nn.Module):
    def __init__(self, n_channels, concept_size, embedding_size, num_neurons = 64, num_layers = 4, input_size = 64, disentangle=False, **kwargs):
        super().__init__()

        self.concept_size = concept_size
        self.embedding_size = embedding_size

        tot_emb_size = concept_size #+ embedding_size

        downs = [DoubleConv(n_channels, num_neurons)]
        for i in range(num_layers):
            downs.append(Down(num_neurons*(2**i), num_neurons*(2**(i+1))))
        self.downs = torch.nn.ModuleList(downs)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()

        encoder_features = decoder_features = num_neurons*(2**num_layers)

        # Encoder MLP
        self.encoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(encoder_features, encoder_features*2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_features*2, tot_emb_size),
        )

        # Decoder MLP
        self.decoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(tot_emb_size, decoder_features*2),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_features*2, decoder_features)
        )

        self.unflatten = torch.nn.Unflatten(1, (decoder_features, 1, 1))

        ups = [torch.nn.Upsample(scale_factor=input_size//(2**num_layers), mode='bilinear', align_corners=True)]
        for i in range(num_layers):
            ups.append(Up2(num_neurons*(2**(num_layers-i)), num_neurons*(2**(num_layers-i-1))))
        ups.append(OutConv(num_neurons, n_channels))
        self.ups = torch.nn.ModuleList(ups)

        self.recon_func = torch.nn.Sigmoid()

        self.concept_func = lambda x: x #torch.sigmoid(x)

        # if disentangle:
        #     self.disentangle_decode_encode = lambda x,y: self.disentangle(x,y)
        # else:
        #     self.disentangle_decode_encode = lambda *args: {}

    def forward(self, x):
        out = self.encode(x)

        out.update(self.decode(out["concepts"], out["embedding"]))

        # out.update(self.disentangle_decode_encode(out["concepts"], out["embedding"]))

        return out
    
    def encode(self, x):
        for down in self.downs:
            x = down(x)

        flat_emb = self.flatten(self.avg_pool(x))

        enc = self.encoder_mlp(flat_emb)

        return {"concepts": self.concept_func(enc)[:, :-self.embedding_size], "embedding": enc[:, -self.embedding_size:]}
    
    def decode(self, concepts, embedding):
        enc = torch.cat([concepts, embedding], dim=-1)
        x = self.unflatten(self.decoder_mlp(enc))
        for up in self.ups:
            x = up(x)
        rec_img = self.recon_func(x)
        
        return {"reconstructed_img": rec_img}
    
    # def disentangle(self, concepts, embedding):
    #     other_concepts = concepts[torch.randperm(concepts.shape[0])]
    #     new_recon_img = self.decode(other_concepts, embedding)["reconstructed_img"]

    #     out = self.encode(new_recon_img)
    #     new_out = {"other_concepts": other_concepts, "new_concepts": out["concepts"], "new_embedding": out["embedding"]}

    #     return new_out

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Up2(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# monocrome --> solo grigio?
# in caso cambiare l'annotazione grigio
    
# class CustomL1Loss(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, input, target):
#         loss = torch.nn.functional.l1_loss(input, target,reduction='none')
#         #sum every dimension, mean over batch
#         loss = loss.sum(dim=1).mean()
#         return 


class LossWithNans(torch.nn.Module):
    def __init__(self, loss, loss_params={}, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.loss = getattr(torch.nn, loss)(**loss_params) if hasattr(torch.nn, loss) else globals()[loss](**loss_params)

    def forward(self, input, target):
        # subset where target is not nan
        mask = torch.isnan(target)
        target[mask] = input[mask]
        return self.loss(input, target)

class AngularL1Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        diff_in_rad = (input-target)*torch.pi*2
        return 3*((1-torch.cos(diff_in_rad))/2)
    
class AngularMSELoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        diff_in_rad = (input-target)*torch.pi*2
        return torch.sqrt(2-2*torch.cos(diff_in_rad))

class KL_DivergenceLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, mu, sigma):
        return 0.5*(sigma**2 + mu**2 - torch.log(sigma**2)-1).mean()

class NormalizedKL_DivergenceLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl = KL_DivergenceLoss()

    def forward(self, mu, sigma):
        kl = self.kl(mu, sigma)
        return 1-torch.exp(-kl)


class TextAIleCustomConceptLoss(torch.nn.Module):
    def __init__(self, separate_losses, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ids = {}
        losses = {}
        for loss_name, (ids,loss_info) in separate_losses.items():
            self.ids[loss_name] = ids
            losses[loss_name] = getattr(torch.nn, loss_info["name"])(**loss_info.get("loss_params",{})) if hasattr(torch.nn, loss_info["name"]) else globals()[loss_info["name"]](**loss_info.get("loss_params",{}))
        self.losses = torch.nn.ModuleDict(losses)

    def forward(self, input, target):
        loss_tot = 0
        for loss_name, loss in self.losses.items():
            inp, trg = input[:,self.ids[loss_name]], target[:,self.ids[loss_name]]
            loss_tot += loss(inp, trg)

        return loss_tot/len(self.losses)
    

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    

class WeightedLoss(torch.nn.Module):
    def __init__(self, loss, loss_params={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = getattr(torch.nn, loss)(**loss_params) if hasattr(torch.nn, loss) else globals()[loss](**loss_params)

    def forward(self, input, target, weight):
        ls = self.loss(input, target)

        wgt = weight.view(len(weight), *(1,)*(len(ls.shape)-1))

        return (wgt*ls).mean()


# Loss: see best permutation of components input/output that gives the minimum loss
class PermutedLoss(torch.nn.Module):
    def __init__(self, loss, loss_params={}, n_components=5, **kwargs):
        super().__init__()

        self.loss = getattr(torch.nn, loss)(reduction="none") if hasattr(torch.nn, loss) else globals()[loss](**loss_params)

        self.permutations = torch.tensor(list(it.permutations(range(n_components))), requires_grad=False)

    def forward(self, input, target):
        # y_hat, y shapes: N, C, X, Y
        # See best permutation of components (C) input/output that gives the minimum loss
        # Produce a C factorial losses matrix
        # C factorial because every permutation of C components is a possible loss
        
        # For each permutation, compute the loss
        losses = torch.zeros((input.shape[0],self.permutations.shape[0]), requires_grad=False, device=input.device)
        for i, permutation in enumerate(self.permutations):
            # Permute the components
            y_hat_permuted = input[:,permutation]
            # Compute the loss
            losses[:,i] = self.loss(y_hat_permuted, target).sum(dim=1)
        
        # Take the minimum loss for each sample
        loss_value = losses.min(1).values.mean()

        return loss_value
    
    #normalize input in 0-1 for each channel
    def normalize_func(self, x):
        mn = x.min(1).values[:,None]
        mx = x.max(1).values[:,None]
        return (x-mn)/(mx-mn+self.eps)
    

