import os
import torch
import torch.nn as nn

from .st_encoder_teacher import PretrainingEncoder
from .st_encoder_student import PretrainingEncoder_student

# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('Weight initial finished!')


def load_moco_encoder_q(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """
    def __init__(self, encoder_teacher, args_encoder, dim=128, K=2048, t=0.07, temp=1e-4):
        """
        dim:        feature dimension (default: 128)
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        base_width: width of the base network
        swav_mlp:   MLP length for SWAV resnet, default=None
        """
        super(SEED, self).__init__()

        self.K = K
        self.t = t
        self.temp = temp
        self.dim = dim
        #self.group = group

        #print("distill hyperparameter queue K", K)
        #print("distill hyperparameter teacher T", temp)
        #print("distill hyperparameter student T", t)

        # create the Teacher/Student encoders
        #self.encoder_teacher = PretrainingEncoder(**args_encoder_teacher)
        self.encoder_teacher = encoder_teacher
        self.encoder_student = PretrainingEncoder_student(**args_encoder)

        #if not args_pretrained:
        #    weights_init(self.encoder_teacher)

        #if args_pretrained:
        #    # freeze all layers but the last fc
        #    for name, param in self.encoder_teacher.named_parameters():
        #        if name not in ['fc.weight', 'fc.bias']:
        #           param.requires_grad = False
        #        else:
        #            print('params', name)

        # load from pre-trained  model
        #load_moco_encoder_q(self.encoder_teacher, args_pretrained)

        # create the queue
        # temporal domain queue
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("tl_queue", torch.randn(dim, K))
        self.tl_queue = nn.functional.normalize(self.tl_queue, dim=0)
        self.register_buffer("tl_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("sl_queue", torch.randn(dim, K))
        self.sl_queue = nn.functional.normalize(self.sl_queue, dim=0)
        self.register_buffer("sl_queue_ptr", torch.zeros(1, dtype=torch.long))


    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, tl_keys, sl_keys):

        N, C = t_keys.shape
        #print(self.K)
        #print(N)

        assert self.K % N == 0  # for simplicity

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K  # move pointer
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K  # move pointer
        self.s_queue_ptr[0] = s_ptr

        sl_ptr = int(self.sl_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.sl_queue[:, sl_ptr:sl_ptr + N] = sl_keys.T
        sl_ptr = (sl_ptr + N) % self.K  # move pointer
        self.sl_queue_ptr[0] = sl_ptr

        tl_ptr = int(self.tl_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.tl_queue[:, tl_ptr:tl_ptr + N] = tl_keys.T
        tl_ptr = (tl_ptr + N) % self.K  # move pointer
        self.tl_queue_ptr[0] = tl_ptr

    #def update_group(self, group):
    #    self.group_buffer = group
    #    #print(self.group_buffer)

    def forward(self, tea_input, stu_input):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """

        #print(self.group_buffer)
        #ll

        # compute query features
        stu_t, stu_s, stu_tl, stu_sl= self.encoder_student(stu_input)  # queries: NxC

        stu_t = nn.functional.normalize(stu_t, dim=1)
        stu_s = nn.functional.normalize(stu_s, dim=1)
        stu_tl = nn.functional.normalize(stu_tl, dim=1)
        stu_sl = nn.functional.normalize(stu_sl, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            tea_t, tea_s, tea_tl, tea_sl = self.encoder_teacher(tea_input)

            tea_t = nn.functional.normalize(tea_t, dim=1)
            tea_s = nn.functional.normalize(tea_s, dim=1)
            tea_tl = nn.functional.normalize(tea_tl, dim=1)
            tea_sl = nn.functional.normalize(tea_sl, dim=1)

        # cross-Entropy Loss
        logit_stu_t = torch.einsum('nc,ck->nk', [stu_t, self.t_queue.clone().detach()])
        logit_stu_s = torch.einsum('nc,ck->nk', [stu_s, self.s_queue.clone().detach()])
        logit_stu_tl = torch.einsum('nc,ck->nk', [stu_tl, self.tl_queue.clone().detach()])
        logit_stu_sl = torch.einsum('nc,ck->nk', [stu_sl, self.sl_queue.clone().detach()])

        logit_tea_t = torch.einsum('nc,ck->nk', [tea_t, self.t_queue.clone().detach()])
        logit_tea_s = torch.einsum('nc,ck->nk', [tea_s, self.s_queue.clone().detach()])
        logit_tea_tl = torch.einsum('nc,ck->nk', [tea_tl, self.tl_queue.clone().detach()])
        logit_tea_sl = torch.einsum('nc,ck->nk', [tea_sl, self.sl_queue.clone().detach()])

        logit_s_t = torch.einsum('nc,nc->n', [stu_t, tea_t]).unsqueeze(-1)
        logit_s_s = torch.einsum('nc,nc->n', [stu_s, tea_s]).unsqueeze(-1)
        logit_s_tl = torch.einsum('nc,nc->n', [stu_tl, tea_tl]).unsqueeze(-1)
        logit_s_sl = torch.einsum('nc,nc->n', [stu_sl, tea_sl]).unsqueeze(-1)

        logit_t_t = torch.einsum('nc,nc->n', [tea_t, tea_t]).unsqueeze(-1)
        logit_t_s = torch.einsum('nc,nc->n', [tea_s, tea_s]).unsqueeze(-1)
        logit_t_tl = torch.einsum('nc,nc->n', [tea_tl, tea_tl]).unsqueeze(-1)
        logit_t_sl = torch.einsum('nc,nc->n', [tea_sl, tea_sl]).unsqueeze(-1)

        logit_stu_1 = torch.cat([logit_s_t, logit_stu_t], dim=1)
        logit_stu_2 = torch.cat([logit_s_s, logit_stu_s], dim=1)
        logit_stu_3 = torch.cat([logit_s_tl, logit_stu_tl], dim=1)
        logit_stu_4 = torch.cat([logit_s_sl, logit_stu_sl], dim=1)

        logit_tea_1 = torch.cat([logit_t_t, logit_tea_t], dim=1)
        logit_tea_2 = torch.cat([logit_t_s, logit_tea_s], dim=1)
        logit_tea_3 = torch.cat([logit_t_tl, logit_tea_tl], dim=1)
        logit_tea_4 = torch.cat([logit_t_sl, logit_tea_sl], dim=1)

        # compute soft labels
        logit_stu_1 /= self.t
        logit_stu_2 /= self.t
        logit_stu_3 /= self.t
        logit_stu_4 /= self.t

        logit_tea_1 = nn.functional.softmax(logit_tea_1 / self.temp, dim=1)
        logit_tea_2 = nn.functional.softmax(logit_tea_2 / self.temp, dim=1)
        logit_tea_3 = nn.functional.softmax(logit_tea_3 / self.temp, dim=1)
        logit_tea_4 = nn.functional.softmax(logit_tea_4 / self.temp, dim=1)

        # de-queue and en-queue
        self._dequeue_and_enqueue(tea_t, tea_s, tea_tl, tea_sl)

        return logit_stu_1, logit_stu_2, logit_stu_3, logit_stu_4, logit_tea_1, logit_tea_2, logit_tea_3, logit_tea_4