import pickle
import random

import numpy as np
import torch
from torch_geometric.utils import to_networkx

from bronstein_paper.base import BaseDataset
from bronstein_paper.gcn import GCN
from bronstein_paper.node_classification import train, evaluate
from bronstein_paper.optimizer import get_optimizer
from bronstein_paper.sdrf import SDRFDataset
from bronstein_paper.seeds import test_seeds, val_seeds
from bronstein_paper.splits import set_train_val_test_split_frac, set_train_val_test_split

if __name__ == '__main__':
    max_steps_Cora_sdrf = 100
    tau_cora_sdrf = 163
    removal_bound_Cora_sdrf = 0.95
    dropout_Cora_sdrf = 0.3396
    hidden_Cora_sdrf = [128]
    lr_Cora_sdrf = 0.0244
    decay_Cora_sdrf = 0.1076

    dropout_Cora_none = 0.4144
    hidden_Cora_none = [64]
    lr_Cora_none = 0.0097
    decay_Cora_none = 0.0639

    dataset = SDRFDataset(max_steps=max_steps_Cora_sdrf, data_dir='dt', undirected=True, remove_edges=True,
                          tau=tau_cora_sdrf, removal_bound=removal_bound_Cora_sdrf)
    # dataset = BaseDataset(undirected=True)
    # with open('dt/processed/Cora_sdrf_ms=100_re=True_rb=0.95_tau=163_lcc=True_undirected.pt', 'rb') as f:
    #     dataset.data = torch.load(f)[0]
    G = to_networkx(dataset.data)
    # print((1754, 721) in G.edges)
    # print((338, 2243) in G.edges)
    # print((926, 1517) in G.edges)
    # print((211, 465) in G.edges)

    # added(1861, 1269)
    # removed(338, 2243)
    # added(1846, 1980)
    # removed(211, 465)

    # val_accs = []
    # vals = []
    # test_accs = []
    # for seed in val_seeds:
    #     data = set_train_val_test_split(seed, dataset.data)
    #     dataset.data = data
    #     model = GCN(dataset, dropout=dropout_Cora_sdrf, hidden=hidden_Cora_sdrf)
    #     optimizer = get_optimizer('adam', model, lr_Cora_sdrf, decay_Cora_sdrf)
    #     # print(1)
    #
    #     best_val = 0
    #     best_test = 0
    #     cur_vals = []
    #     for epoch in range(1001):
    #         # 81.2
    #         loss = train(model, optimizer, data)
    #         # if epoch % 100 == 0:
    #             # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #         # print(loss)
    #         ed = evaluate(model, data, test=True)
    #         val = ed['val_acc']
    #         cur_vals.append(val)
    #     test = ed['test_acc']
    #         # if val > best_val:
    #         #     best_epoch = epoch
    #         #     best_val = val
    #         # if test > best_test:
    #         #     best_test = test
    #     # vals.append(cur_vals)
    #     print('val', val)
    #     val_accs.append(val)
    #     test_accs.append(test)
    #     # print('test', best_test)
    #     # test_accs.append(best_test)
    # # with open('vals', 'wb') as f:
    # #     pickle.dump(vals, f)
    # # print(np.array(vals).mean(axis=0))
    # print(val_accs)
    # print(test_accs)
    # # print(test_accs)
    # print(np.mean(val_accs))
    # print(np.mean(test_accs))
    # biased test set with rewiring
    # [0.8536764705882353, 0.8522058823529411, 0.8492647058823529, 0.85, 0.850735294117647, 0.8492647058823529,
    #  0.850735294117647, 0.8485294117647059, 0.85, 0.8492647058823529, 0.8514705882352941, 0.8522058823529411,
    #  0.850735294117647, 0.8544117647058823, 0.8485294117647059, 0.8477941176470588, 0.85, 0.8544117647058823,
    #  0.8485294117647059, 0.8492647058823529, 0.85, 0.85, 0.8485294117647059, 0.850735294117647, 0.85,
    #  0.8492647058823529, 0.85, 0.8558823529411764, 0.8536764705882353, 0.8477941176470588, 0.850735294117647,
    #  0.8455882352941176, 0.8529411764705882, 0.850735294117647, 0.8514705882352941, 0.8522058823529411,
    #  0.8492647058823529, 0.8492647058823529, 0.85, 0.8529411764705882, 0.8492647058823529, 0.8463235294117647,
    #  0.8485294117647059, 0.850735294117647, 0.8492647058823529, 0.8529411764705882, 0.850735294117647,
    #  0.8529411764705882, 0.8522058823529411, 0.8477941176470588, 0.8470588235294118, 0.8514705882352941,
    #  0.8477941176470588, 0.850735294117647, 0.8536764705882353, 0.85, 0.8463235294117647, 0.8485294117647059, 0.85,
    #  0.8485294117647059, 0.8514705882352941, 0.850735294117647, 0.8522058823529411, 0.8492647058823529,
    #  0.8477941176470588, 0.8536764705882353, 0.8485294117647059, 0.8522058823529411, 0.85, 0.8522058823529411,
    #  0.8485294117647059, 0.8485294117647059, 0.8455882352941176, 0.8522058823529411, 0.8492647058823529,
    #  0.8522058823529411, 0.8514705882352941, 0.8477941176470588, 0.8485294117647059, 0.850735294117647,
    #  0.8463235294117647, 0.8477941176470588, 0.8514705882352941, 0.8551470588235294, 0.8544117647058823,
    #  0.850735294117647, 0.8492647058823529, 0.8529411764705882, 0.8492647058823529, 0.850735294117647, 0.85,
    #  0.8522058823529411, 0.8485294117647059, 0.8522058823529411, 0.8492647058823529, 0.8529411764705882, 0.85,
    #  0.8536764705882353, 0.8514705882352941, 0.8477941176470588]
    # [0.8274111675126904, 0.8304568527918782, 0.8385786802030457, 0.8345177664974619, 0.8345177664974619,
    #  0.8324873096446701, 0.833502538071066, 0.8345177664974619, 0.8304568527918782, 0.8324873096446701,
    #  0.8324873096446701, 0.8375634517766497, 0.8304568527918782, 0.8314720812182741, 0.8324873096446701,
    #  0.8375634517766497, 0.833502538071066, 0.8324873096446701, 0.833502538071066, 0.8375634517766497,
    #  0.8304568527918782, 0.8314720812182741, 0.8355329949238579, 0.8284263959390863, 0.8284263959390863,
    #  0.8324873096446701, 0.8304568527918782, 0.8345177664974619, 0.8304568527918782, 0.8314720812182741,
    #  0.8294416243654822, 0.8294416243654822, 0.8365482233502538, 0.8355329949238579, 0.8345177664974619,
    #  0.8314720812182741, 0.8324873096446701, 0.8304568527918782, 0.8365482233502538, 0.8314720812182741,
    #  0.833502538071066, 0.8314720812182741, 0.8345177664974619, 0.8324873096446701, 0.8274111675126904,
    #  0.8345177664974619, 0.833502538071066, 0.8375634517766497, 0.8304568527918782, 0.8304568527918782,
    #  0.8294416243654822, 0.8284263959390863, 0.8294416243654822, 0.8314720812182741, 0.8324873096446701,
    #  0.833502538071066, 0.8324873096446701, 0.833502538071066, 0.8324873096446701, 0.8324873096446701,
    #  0.8314720812182741, 0.8355329949238579, 0.8274111675126904, 0.8274111675126904, 0.8263959390862944,
    #  0.8314720812182741, 0.8355329949238579, 0.8355329949238579, 0.8324873096446701, 0.8294416243654822,
    #  0.8304568527918782, 0.8314720812182741, 0.8274111675126904, 0.8314720812182741, 0.8416243654822335,
    #  0.8324873096446701, 0.8263959390862944, 0.833502538071066, 0.8355329949238579, 0.8304568527918782,
    #  0.8355329949238579, 0.8324873096446701, 0.8365482233502538, 0.8304568527918782, 0.8314720812182741,
    #  0.8314720812182741, 0.8284263959390863, 0.8274111675126904, 0.8345177664974619, 0.8314720812182741,
    #  0.8345177664974619, 0.8314720812182741, 0.8345177664974619, 0.833502538071066, 0.8274111675126904,
    #  0.8375634517766497, 0.8324873096446701, 0.8395939086294416, 0.8294416243654822, 0.8314720812182741]
    # 0.8503602941176469
    # 0.8323654822335027

    # biased test set without rewiring
    # [0.8316176470588236, 0.8139705882352941, 0.8279411764705882, 0.8544117647058823, 0.8382352941176471, 0.85,
    #  0.8213235294117647, 0.8169117647058823, 0.8463235294117647, 0.8308823529411765, 0.8419117647058824,
    #  0.8154411764705882, 0.8411764705882353, 0.8419117647058824, 0.8382352941176471, 0.8272058823529411,
    #  0.8338235294117647, 0.8441176470588235, 0.8286764705882353, 0.8235294117647058, 0.8522058823529411,
    #  0.8455882352941176, 0.8375, 0.8470588235294118, 0.8125, 0.8397058823529412, 0.8411764705882353, 0.8279411764705882,
    #  0.8404411764705882, 0.8448529411764706, 0.8316176470588236, 0.8338235294117647, 0.8375, 0.8213235294117647,
    #  0.8110294117647059, 0.8323529411764706, 0.8132352941176471, 0.8323529411764706, 0.8198529411764706,
    #  0.861764705882353, 0.8514705882352941, 0.8360294117647059, 0.8389705882352941, 0.8279411764705882,
    #  0.8279411764705882, 0.8492647058823529, 0.8316176470588236, 0.8382352941176471, 0.8286764705882353,
    #  0.8029411764705883, 0.8286764705882353, 0.8272058823529411, 0.8375, 0.8404411764705882, 0.8352941176470589, 0.8375,
    #  0.8411764705882353, 0.8301470588235295, 0.8264705882352941, 0.8169117647058823, 0.8073529411764706,
    #  0.8301470588235295, 0.8264705882352941, 0.8338235294117647, 0.8330882352941177, 0.8161764705882353,
    #  0.8102941176470588, 0.825735294117647, 0.8316176470588236, 0.8352941176470589, 0.8051470588235294,
    #  0.8485294117647059, 0.8397058823529412, 0.8183823529411764, 0.8397058823529412, 0.8132352941176471,
    #  0.8588235294117647, 0.825735294117647, 0.8397058823529412, 0.8352941176470589, 0.8286764705882353,
    #  0.8227941176470588, 0.836764705882353, 0.8294117647058824, 0.8169117647058823, 0.8338235294117647,
    #  0.8205882352941176, 0.8169117647058823, 0.8213235294117647, 0.8272058823529411, 0.8360294117647059,
    #  0.8360294117647059, 0.825735294117647, 0.8404411764705882, 0.8191176470588235, 0.8404411764705882,
    #  0.8139705882352941, 0.8455882352941176, 0.8220588235294117, 0.8220588235294117]
    # [0.817258883248731, 0.7928934010152284, 0.8233502538071066, 0.8355329949238579, 0.8304568527918782,
    #  0.8446700507614213, 0.8131979695431472, 0.8223350253807107, 0.8314720812182741, 0.8233502538071066,
    #  0.8223350253807107, 0.8060913705583757, 0.8233502538071066, 0.8395939086294416, 0.8233502538071066,
    #  0.8182741116751269, 0.8213197969543147, 0.8314720812182741, 0.8050761421319796, 0.8253807106598985,
    #  0.849746192893401, 0.8517766497461929, 0.8314720812182741, 0.8416243654822335, 0.8263959390862944,
    #  0.8385786802030457, 0.8446700507614213, 0.8081218274111676, 0.8294416243654822, 0.8517766497461929,
    #  0.8152284263959391, 0.8263959390862944, 0.8416243654822335, 0.833502538071066, 0.8162436548223351,
    #  0.8324873096446701, 0.8111675126903554, 0.8294416243654822, 0.8182741116751269, 0.8467005076142132,
    #  0.8294416243654822, 0.8223350253807107, 0.8152284263959391, 0.8131979695431472, 0.8294416243654822,
    #  0.8406091370558376, 0.8203045685279188, 0.8253807106598985, 0.817258883248731, 0.8131979695431472,
    #  0.8365482233502538, 0.8253807106598985, 0.8213197969543147, 0.8304568527918782, 0.8233502538071066,
    #  0.8233502538071066, 0.8284263959390863, 0.8233502538071066, 0.8304568527918782, 0.8162436548223351,
    #  0.7979695431472081, 0.8162436548223351, 0.8223350253807107, 0.8152284263959391, 0.8365482233502538,
    #  0.8162436548223351, 0.8192893401015229, 0.8040609137055837, 0.817258883248731, 0.8314720812182741,
    #  0.8091370558375635, 0.8395939086294416, 0.8284263959390863, 0.8213197969543147, 0.8375634517766497,
    #  0.8131979695431472, 0.8477157360406091, 0.8304568527918782, 0.8284263959390863, 0.8263959390862944,
    #  0.8121827411167513, 0.7979695431472081, 0.8446700507614213, 0.8213197969543147, 0.8274111675126904,
    #  0.817258883248731, 0.817258883248731, 0.8284263959390863, 0.8162436548223351, 0.8385786802030457,
    #  0.8162436548223351, 0.8365482233502538, 0.8243654822335026, 0.8345177664974619, 0.8243654822335026,
    #  0.8395939086294416, 0.8182741116751269, 0.8253807106598985, 0.8213197969543147, 0.833502538071066]
    # 0.8313602941176471
    # 0.8253502538071064

    # unbiased no rewiring
    # [0.8286764705882353, 0.8154411764705882, 0.8279411764705882, 0.8588235294117647, 0.8419117647058824,
    #  0.8485294117647059, 0.8198529411764706, 0.8117647058823529, 0.8441176470588235, 0.8264705882352941,
    #  0.8397058823529412, 0.8191176470588235, 0.8397058823529412, 0.8411764705882353, 0.8411764705882353,
    #  0.8301470588235295, 0.8338235294117647, 0.8426470588235294, 0.8279411764705882, 0.8213235294117647,
    #  0.8477941176470588, 0.8441176470588235, 0.8411764705882353, 0.8463235294117647, 0.8176470588235294,
    #  0.8426470588235294, 0.836764705882353, 0.8227941176470588, 0.8301470588235295, 0.8477941176470588,
    #  0.8345588235294118, 0.8338235294117647, 0.836764705882353, 0.8242647058823529, 0.8139705882352941,
    #  0.8308823529411765, 0.8161764705882353, 0.8360294117647059, 0.8301470588235295, 0.8661764705882353,
    #  0.8485294117647059, 0.836764705882353, 0.8375, 0.8345588235294118, 0.8242647058823529, 0.8485294117647059, 0.825,
    #  0.836764705882353, 0.8294117647058824, 0.8110294117647059, 0.8294117647058824, 0.825, 0.8397058823529412,
    #  0.8404411764705882, 0.8316176470588236, 0.8448529411764706, 0.8448529411764706, 0.8272058823529411,
    #  0.8242647058823529, 0.8205882352941176, 0.799264705882353, 0.8345588235294118, 0.8198529411764706,
    #  0.8227941176470588, 0.8272058823529411, 0.8161764705882353, 0.8169117647058823, 0.8205882352941176,
    #  0.8323529411764706, 0.8330882352941177, 0.8022058823529412, 0.850735294117647, 0.8382352941176471,
    #  0.8176470588235294, 0.8463235294117647, 0.8117647058823529, 0.8588235294117647, 0.825, 0.8382352941176471,
    #  0.8294117647058824, 0.8279411764705882, 0.8205882352941176, 0.8338235294117647, 0.8308823529411765,
    #  0.8154411764705882, 0.8279411764705882, 0.8264705882352941, 0.8220588235294117, 0.8205882352941176,
    #  0.825735294117647, 0.8389705882352941, 0.8352941176470589, 0.825, 0.8411764705882353, 0.8205882352941176,
    #  0.8411764705882353, 0.8161764705882353, 0.8382352941176471, 0.8272058823529411, 0.8227941176470588]
    # [0.8101522842639594, 0.7928934010152284, 0.8162436548223351, 0.8304568527918782, 0.8213197969543147,
    #  0.8304568527918782, 0.8040609137055837, 0.8071065989847716, 0.8253807106598985, 0.8091370558375635,
    #  0.8203045685279188, 0.8050761421319796, 0.8101522842639594, 0.8304568527918782, 0.8131979695431472,
    #  0.8060913705583757, 0.8142131979695432, 0.8111675126903554, 0.7888324873096447, 0.8121827411167513,
    #  0.8375634517766497, 0.8406091370558376, 0.8294416243654822, 0.8365482233502538, 0.8213197969543147,
    #  0.8274111675126904, 0.8274111675126904, 0.8, 0.8233502538071066, 0.8416243654822335, 0.8020304568527918,
    #  0.8233502538071066, 0.8365482233502538, 0.8274111675126904, 0.8152284263959391, 0.8182741116751269,
    #  0.8081218274111676, 0.8182741116751269, 0.8142131979695432, 0.8416243654822335, 0.8203045685279188,
    #  0.8213197969543147, 0.8091370558375635, 0.8233502538071066, 0.817258883248731, 0.8233502538071066,
    #  0.8071065989847716, 0.817258883248731, 0.8060913705583757, 0.8213197969543147, 0.8253807106598985,
    #  0.8020304568527918, 0.8213197969543147, 0.8243654822335026, 0.8192893401015229, 0.8324873096446701,
    #  0.817258883248731, 0.8243654822335026, 0.8162436548223351, 0.8101522842639594, 0.7898477157360406,
    #  0.8060913705583757, 0.8203045685279188, 0.8060913705583757, 0.8284263959390863, 0.8030456852791878,
    #  0.8050761421319796, 0.7969543147208121, 0.8050761421319796, 0.8182741116751269, 0.8091370558375635,
    #  0.8294416243654822, 0.8131979695431472, 0.8152284263959391, 0.8294416243654822, 0.8020304568527918,
    #  0.8467005076142132, 0.8253807106598985, 0.8091370558375635, 0.8152284263959391, 0.8030456852791878,
    #  0.782741116751269, 0.8274111675126904, 0.8101522842639594, 0.8243654822335026, 0.8091370558375635,
    #  0.8152284263959391, 0.8131979695431472, 0.8071065989847716, 0.8314720812182741, 0.7979695431472081,
    #  0.8223350253807107, 0.8121827411167513, 0.8203045685279188, 0.8060913705583757, 0.8446700507614213,
    #  0.8131979695431472, 0.8071065989847716, 0.8091370558375635, 0.8274111675126904]
    # 0.8311985294117648
    # 0.8166700507614215
