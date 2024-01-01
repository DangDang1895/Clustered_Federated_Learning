import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

#用于记录实验结果的简单日志类。它具有一个log方法，
class ExperimentLogger:
    #用于将键值对记录到实例的属性中。如果键不存在，则创建一个新的属性并将值存储为列表；如果键已经存在，则将值添加到相应的列表中。
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]


def show_acc(client_stats,communication_rounds,MobileNetV2_EPS_1,MobileNetV2_EPS_2,ResNet8_EPS_1,ResNet8_EPS_2):
    clear_output(wait=True)
    plt.figure(figsize=(20,20))

    plt.subplot(4,3,1)
    acc_mean = np.mean(client_stats.MobileNetV2_loss, axis=1)
    acc_std = np.std(client_stats.MobileNetV2_loss, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_MobileNetV2[-1]]))

    plt.xlabel("Communication Rounds")
    plt.ylabel("loss")
    plt.title("MobileNetV2_loss")
    plt.xlim(0, communication_rounds)

    plt.subplot(4,3,2)
    acc_mean = np.mean(client_stats.MobileNetV2_local_acc, axis=1)
    acc_std = np.std(client_stats.MobileNetV2_local_acc, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_MobileNetV2[-1]]))

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2_local_traindata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(4,3,3)
    acc_mean = np.mean(client_stats.MobileNetV2_local_test_acc, axis=1)
    acc_std = np.std(client_stats.MobileNetV2_local_test_acc, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_MobileNetV2[-1]]))

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2_local_testdata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)


    plt.subplot(4,3,4)
    acc_mean = np.mean(client_stats.clients_train_MobileNetV2, axis=1)
    acc_std = np.std(client_stats.clients_train_MobileNetV2, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_MobileNetV2[-1]]))

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2_global_traindata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)



    plt.subplot(4,3,5)
    acc_mean = np.mean(client_stats.clients_acc_MobileNetV2, axis=1)
    acc_std = np.std(client_stats.clients_acc_MobileNetV2, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_MobileNetV2[-1]]))

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2_global_testdata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(4,3,6)
    
    plt.plot(client_stats.rounds, client_stats.MobileNetV2_mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(client_stats.rounds, client_stats.MobileNetV2_max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    plt.axhline(y=MobileNetV2_EPS_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=MobileNetV2_EPS_2, linestyle=":", color="k", label=r"$\varepsilon_2$")

    if "split_MobileNetV2" in client_stats.__dict__:
        for s in client_stats.split_MobileNetV2:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")

    plt.xlabel("Communication Rounds")
    plt.legend()   
    plt.xlim(0, communication_rounds)
 

    plt.subplot(4,3,7)
    acc_mean = np.mean(client_stats.ResNet8_loss, axis=1)
    acc_std = np.std(client_stats.ResNet8_loss, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_ResNet8[-1]]))


    plt.xlabel("Communication Rounds")
    plt.ylabel("loss")
    plt.title("ResNet8_loss")
    plt.xlim(0, communication_rounds)


    plt.subplot(4,3,8)
    acc_mean = np.mean(client_stats.ResNet8_local_acc, axis=1)
    acc_std = np.std(client_stats.ResNet8_local_acc, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_ResNet8[-1]]))


    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("ResNet8_local_traindata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(4,3,9)
    acc_mean = np.mean(client_stats.ResNet8_local_test_acc, axis=1)
    acc_std = np.std(client_stats.ResNet8_local_test_acc, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_ResNet8[-1]]))


    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("ResNet8_local_testdata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)



    plt.subplot(4,3,10)
    acc_mean = np.mean(client_stats.clients_train_ResNet8, axis=1)
    acc_std = np.std(client_stats.clients_train_ResNet8, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    #plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_ResNet8[-1]]))


    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("ResNet8_global_traindata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(4,3,11)
    acc_mean = np.mean(client_stats.clients_acc_ResNet8, axis=1)
    acc_std = np.std(client_stats.clients_acc_ResNet8, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    plt.text(x=communication_rounds, y=1, ha="right", va="top", s="Clusters: {}".format([x for x in client_stats.cluster_ResNet8[-1]]))


    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("ResNet8_global_testdata_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.subplot(4,3,12)
    
    plt.plot(client_stats.rounds, client_stats.ResNet8_mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(client_stats.rounds, client_stats.ResNet8_max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    plt.axhline(y=ResNet8_EPS_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=ResNet8_EPS_2, linestyle=":", color="k", label=r"$\varepsilon_2$")

    if "split_ResNet8" in client_stats.__dict__:
        for s in client_stats.split_ResNet8:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")

    plt.xlabel("Communication Rounds")
    plt.legend()
    plt.xlim(0, communication_rounds)

    plt.show()
