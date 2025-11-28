import torch
import torch.nn as nn

from .st_encoder_pretraining import PretrainingEncoder

# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')

class ST_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ST_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        self.encoder_q = PretrainingEncoder(**args_encoder) # query encoder
        self.encoder_k = PretrainingEncoder(**args_encoder) # key encoder
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

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

        # instance level queue
        self.register_buffer("td_queue", torch.randn(dim, K))
        self.td_queue = nn.functional.normalize(self.td_queue, dim=0)
        self.register_buffer("td_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("sd_queue", torch.randn(dim, K))
        self.sd_queue = nn.functional.normalize(self.sd_queue, dim=0)
        self.register_buffer("sd_queue_ptr", torch.zeros(1, dtype=torch.long))

        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, tl_keys, sl_keys, td_keys, sd_keys):
        N, C = t_keys.shape

        assert self.K % N == 0  # for simplicity

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T   #行
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

        sd_ptr = int(self.sd_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.sd_queue[:, sd_ptr:sd_ptr + N] = sd_keys.T
        sd_ptr = (sd_ptr + N) % self.K  # move pointer
        self.sd_queue_ptr[0] = sd_ptr

        td_ptr = int(self.td_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.td_queue[:, td_ptr:td_ptr + N] = td_keys.T
        td_ptr = (td_ptr + N) % self.K  # move pointer
        self.td_queue_ptr[0] = td_ptr

    def forward(self, q_input, k_input):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """
        # compute temporal domain level, spatial domain level and instance level features
        qt, qs, qtl, qsl, qtd, qsd = self.encoder_q(q_input)  # queries: NxC

        qt = nn.functional.normalize(qt, dim=1) #global temporal
        qs = nn.functional.normalize(qs, dim=1) #global spatial
        qsl = nn.functional.normalize(qsl, dim=1) #local spatial
        qtl = nn.functional.normalize(qtl, dim=1) #local temporal
        qsd = nn.functional.normalize(qsd, dim=1) #combination spatial
        qtd = nn.functional.normalize(qtd, dim=1) #combination temporal

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            kt, ks, ktl, ksl, ktd, ksd= self.encoder_k(k_input)  # keys: NxC

            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ktl = nn.functional.normalize(ktl, dim=1)
            ksl = nn.functional.normalize(ksl, dim=1)
            ktd = nn.functional.normalize(ktd, dim=1)
            ksd = nn.functional.normalize(ksd, dim=1)

        # interactive loss
        #正样本相似度
        l_pos_1 = torch.einsum('nc,nc->n', [qs, ksl]).unsqueeze(1)
        l_pos_2 = torch.einsum('nc,nc->n', [qsl, ks]).unsqueeze(1)
        l_pos_3 = torch.einsum('nc,nc->n', [qt, ktl]).unsqueeze(1)
        l_pos_4 = torch.einsum('nc,nc->n', [qtl, kt]).unsqueeze(1)

        l_pos_5 = torch.einsum('nc,nc->n', [qtd, ksd]).unsqueeze(1)
        l_pos_6 = torch.einsum('nc,nc->n', [qsd, ktd]).unsqueeze(1)
        #负样本相似度
        l_neg_1 = torch.einsum('nc,ck->nk', [qs, self.sl_queue.clone().detach()])
        l_neg_2 = torch.einsum('nc,ck->nk', [qsl, self.s_queue.clone().detach()])
        l_neg_3 = torch.einsum('nc,ck->nk', [qt, self.tl_queue.clone().detach()])
        l_neg_4 = torch.einsum('nc,ck->nk', [qtl, self.t_queue.clone().detach()])

        l_neg_5 = torch.einsum('nc,ck->nk', [qtd, self.sd_queue.clone().detach()])
        l_neg_6 = torch.einsum('nc,ck->nk', [qsd, self.td_queue.clone().detach()])
        #拼接正负样本：构造 logits
        logits_1 = torch.cat([l_pos_1, l_neg_1], dim=1)    #n x (1+K)
        logits_2 = torch.cat([l_pos_2, l_neg_2], dim=1)    #n x (1+K)
        logits_3 = torch.cat([l_pos_3, l_neg_3], dim=1)
        logits_4 = torch.cat([l_pos_4, l_neg_4], dim=1)

        logits_5 = torch.cat([l_pos_5, l_neg_5], dim=1)
        logits_6 = torch.cat([l_pos_6, l_neg_6], dim=1)
        # apply temperature
        logits_1 /= self.T
        logits_2 /= self.T
        logits_3 /= self.T
        logits_4 /= self.T

        logits_5 /= self.T
        logits_6 /= self.T

        labels_1 = torch.zeros(logits_1.shape[0], dtype=torch.long).cuda()
        labels_2 = torch.zeros(logits_2.shape[0], dtype=torch.long).cuda()
        labels_3 = torch.zeros(logits_3.shape[0], dtype=torch.long).cuda()
        labels_4 = torch.zeros(logits_4.shape[0], dtype=torch.long).cuda()
        
        labels_5 = torch.zeros(logits_5.shape[0], dtype=torch.long).cuda()
        labels_6 = torch.zeros(logits_6.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(kt, ks, ktl, ksl, ktd, ksd)

        return logits_1, logits_2, logits_3, logits_4, logits_5, logits_6,\
            labels_1, labels_2, labels_3, labels_4, labels_5, labels_6