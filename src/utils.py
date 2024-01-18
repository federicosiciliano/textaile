import torch
import torchvision

# Concept-based AutoEncoder
class TextAIle(torch.nn.Module):
    def __init__(self, encoder, decoder, embedding_size, vae = False, disentagle = False, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_size = embedding_size

        self.concept_func = torch.nn.Sigmoid()
        self.recon_func = torch.nn.Sigmoid()

        if vae:
            self.linear_mu = torch.nn.Linear(embedding_size, embedding_size)
            self.linear_sigma = torch.nn.Linear(embedding_size, embedding_size)
            self.distribution = torch.normal
            self.vae_encode = lambda x: self.vae(x)
        else:
            self.vae_encode = lambda x: {"embedding": x}

        if disentagle:
            self.disentagle_decode_encode = lambda x,y: self.disentagle(x,y)
        else:
            self.disentagle_decode_encode = lambda x,y: {}

        # parametri dropout concepts

    def forward(self, x):
        #grayscale_x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=3)

        out = self.encode(x)

        # dropout_concepts --> concepts
        # [dropout_concepts, embedding] --> concepts

        # dropout_concepts = torch.nn.Dropout(p=0.2)(concepts)*10
        # Dropout not applied on first three components and last 6 components
        # ...
        
        #[a,b] --> [a+1,b+1]

        # rec_concepts = self.emb_to_concepts_nn(embedding)
        # Loss(concepts, rec_concepts) --> min

        # Autoencoder --> max Loss(concepts, rec_concepts)

        # Aggiungere pezzo loss recon_concepts

        # Reconstruct input
        out.update(self.decode(torch.cat([out["concepts"], out["embedding"]], dim=-1)))

        # Disentangle
        out.update(self.disentagle_decode_encode(out["concepts"], out["embedding"]))

        return out
    
    def encode(self, x):
        enc = self.encoder(x)
        concepts = self.concept_func(enc[:, :-self.embedding_size])
        embedding = enc[:, -self.embedding_size:]
        out = self.vae_encode(embedding)
        out["concepts"] = concepts
        return out

    def decode(self, x):
        return {"reconstructed_img": self.recon_func(self.decoder(x))}
    
    def vae(self, x):
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5*self.linear_sigma(x))+1e-8
        z = self.distribution(mu, sigma)
        return {"embedding": z, "mu": mu, "sigma": sigma}
    
    def disentagle(self, concepts, embedding):
        # image --> (concepts)/embedding --> (random_concepts) / embedding--> rec_image --> (new_concepts)/new_embedding
        # #x_recon2 = self.decode(torch.cat([torch.rand_like(concepts), embedding], dim=-1)) #se no permutazione sul batch di concepts
        random_concepts = torch.rand_like(concepts)
        new_complete_embedding = torch.cat([random_concepts, embedding], dim=-1)
        new_recon_img = self.decode(new_complete_embedding)["reconstructed_img"]

        out = self.encode(new_recon_img)
        new_out = {"random_concepts": random_concepts, "new_embedding": out["embedding"], "new_concepts": out["concepts"]}

        return new_out

        # random_concepts == new_concepts
        ### new_concepts != concepts
        # new_embedding == embedding


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
        mask = ~torch.isnan(target)
        return self.loss(input[mask], target[mask])

class AngularL1Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        diff_in_rad = (input-target)*torch.pi*2
        return ((1-torch.cos(diff_in_rad))/2)
    
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
        # print("MU:", mu)
        # print("SIGMA:", sigma)
        return 0.5*(sigma**2 + mu**2 - torch.log(sigma**2)-1).mean()

class NormalizedKL_DivergenceLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl = KL_DivergenceLoss()

    def forward(self, mu, sigma):
        kl = self.kl(mu, sigma)
        return 1-torch.exp(-kl)


class TextAIleCustomConceptLoss(torch.nn.Module):
    def __init__(self, angular_ids, angular_loss, other_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.angular_ids = angular_ids
        self.angular_loss = getattr(torch.nn, angular_loss["name"])(**angular_loss.get("params",{})) if hasattr(torch.nn, angular_loss["name"]) else globals()[angular_loss["name"]](**angular_loss.get("params",{}))
        self.other_loss = getattr(torch.nn, other_loss["name"])(**other_loss.get("params",{})) if hasattr(torch.nn, other_loss["name"]) else globals()[other_loss["name"]](**other_loss.get("params",{}))
    
    def forward(self, input, target):
        # Separate angular values
        input_ang = input[:, self.angular_ids]
        target_ang = target[:, self.angular_ids]
        # subset to non nans
        mask = ~torch.isnan(target_ang)
        input_ang = input_ang[mask]
        target_ang = target_ang[mask]

        # Separate other values
        mask = torch.ones(input.shape, dtype=bool)
        mask[:, self.angular_ids] = False
        input_other = input[mask]
        target_other = target[mask]

        # Compute loss
        loss_ang = self.angular_loss(input_ang, target_ang)
        loss_other = self.other_loss(input_other, target_other)

        loss_tot = torch.cat([loss_ang, loss_other], dim=-1)
        return loss_tot.mean()
    

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