import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from numpy import interp
from sklearn.metrics import auc

# #############################################################################
def draw_mean_roc(path, color, model_name):
    with open(path+"fpr.csv", "rb") as f:
        fpr_list = pickle.load(f)
    with open(path+"tpr.csv", "rb") as f:
        tpr_list = pickle.load(f)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for j in range(len(fpr_list)):
        fpr, tpr = fpr_list[j], tpr_list[j]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ## TODO the auc should be compute by new tpr&fpr or mean aucs?
    ## i choose new tpr&fpr, because the curve is drawed by new tpr&fpr, the mean auc should be consistent with the curve.
    plt.plot(mean_fpr, mean_tpr, color=color, label=r'%s (AUC=%0.4f$\pm$%0.4f)' % (model_name, mean_auc, std_auc), lw=2, alpha=.8)


# #############################################################################
def draw_mean_prc(path, color, model_name):
    with open(path+"precision_point.csv", "rb") as f:
        precision_point_list = pickle.load(f)
    with open(path+"recall_point.csv", "rb") as f:
        recall_point_list = pickle.load(f)

    mean_precision_list = list()
    mean_average_precision = list()
    mean_recall = np.linspace(0, 1, 101)

    for j in range(len(precision_point_list)):

        precision, recall = precision_point_list[j], recall_point_list[j]
        # print(j, precision.shape,recall.shape)
        
        average_precision = auc(recall, precision)
        mean_average_precision.append(average_precision)

        # plt.plot(precision, recall, color=color, alpha=0.05)

        precision = np.interp(mean_recall, precision, recall)
        # precision[0] = 0.0
        mean_precision_list.append(precision)

    mean_precision_list = np.array(mean_precision_list)
    mean_precision = mean_precision_list.mean(axis=0)

    plt.plot(mean_precision, mean_recall, label=r"%s (AUC=%.2f$\pm$%0.2f)" % (model_name, auc(mean_recall, mean_precision), np.std(mean_average_precision)), color=color)

np.random.seed(9523)

FONTSIZE = 10
# plt.figure(dpi=300, figsize=(8, 10))
plt.figure(dpi=300, figsize=(8, 13))
# plt.style.use("fast")
plt.rc("font", family="Times New Roman")
params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": 5,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
plt.rcParams.update(params)

x = list(range(0, 200, 10))
x_np = np.array(x)
crispr_m_auprc_circle_guide = [0.37730942421062935, 0.3729380449206421, 0.42403189981058464, 0.3417424162991369, 0.3624040209651558, 0.41460206979880415, 0.39836547075575, 0.3808828358434153, 0.34379232067521215, 0.39578375408860905, 0.38933350523670357, 0.3958215077818623, 0.44914222117493574, 0.42812085259082944, 0.33321732213688, 0.39539340561811404, 0.40167147562212246, 0.39213533678103546, 0.3840662098758275, 0.4164702868302576]
r_crispr_auprc_circle_guide = [0.33117429718016783, 0.21455258346940903, 0.09025204757664579, 0.11888460283356583, 0.21874036659210283, 0.4047313131710089, 0.07244575089186128, 0.1393304993646361, 0.08701047244856952, 0.1654464893447289, 0.1645871244993349, 0.19117429083229545, 0.21847257732623573, 0.4026079222925245, 0.11986307012944003, 0.20945326246708595, 0.3397323339236034, 0.2931391317178899, 0.13884357578672726, 0.2772898430955618]
crispr_ip_auprc_circle_guide = [0.13655032275233175, 0.1292645854275888, 0.18961923950980936, 0.24774982108678933, 0.15051415959760803, 0.09276663392471578, 0.16757734555359816, 0.1631378553590275, 0.14519691998962572, 0.235603380371181, 0.16029166040886458, 0.15318274910265764, 0.14822835519365446, 0.18518946109188786, 0.16717432207334476, 0.17829810808764712, 0.18930147361538815, 0.16768902069958133, 0.1644241459330014, 0.11988928013767042]
crispr_net_auprc_circle_guide = [0.15481368919013899, 0.24680420415934504, 0.2939706752474063, 0.08515117636930646, 0.16569330890393804, 0.14892130365524354, 0.2722002322351067, 0.17101904326072837, 0.28176208241940708, 0.20459389126835776, 0.31281805678462705, 0.17659229966289512, 0.352280920594497, 0.15002089407518038, 0.2711536035382024, 0.23083161146849697, 0.16784647239735676, 0.08969510594463907, 0.15323233777237755, 0.16298150940949434]

def get_fitted_y(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    y_fit = poly(x)
    return y_fit

degree = 9

# * fig(a)
print("ploting fig(a)...")
plt.subplot(4, 2, 1)
# plt.plot(x, crispr_m, color='#ED1C24', label='CRISPR-M')
y_fit = get_fitted_y(x_np, crispr_m_auprc_circle_guide, degree)
plt.plot(x_np, y_fit, color='#ED1C24', label='CRISPR-M')
# plt.plot(x, r_crispr_auprc, color='#22B14C', label='R-CRISPR')
y_fit = get_fitted_y(x_np, r_crispr_auprc_circle_guide, degree)
plt.plot(x_np, y_fit, color='#22B14C', label='R-CRISPR')
# plt.plot(x, crispr_ip_auprc, color='#00A2E8', label='CRISPR-IP')
y_fit = get_fitted_y(x_np, crispr_ip_auprc_circle_guide, degree)
plt.plot(x_np, y_fit, color='#00A2E8', label='CRISPR-IP')
# plt.plot(x, crispr_net_auprc, color='#FFC90E', label='CRISPR-NET')
y_fit = get_fitted_y(x_np, crispr_net_auprc_circle_guide, degree)
plt.plot(x_np, y_fit, color='#FFC90E', label='CRISPR-NET')

plt.xlabel('Random seed')
plt.ylabel('AUPRC')
plt.legend(loc="best")
plt.xlim(-5, 195)
plt.title('(a) Impact of Random Seed on AUPRC\non CIRCLE_GUIDE Dataset')

# * fig(b)
print("ploting fig(b)...")
plt.subplot(4, 2, 2)
crispr_m_auprc_mismatch = [0.43564634372779075, 0.4327428072149392, 0.4360368944057097, 0.42613537153319005, 0.4373442578876573, 0.42770719289789185, 0.42871859299738655, 0.4374726196485207, 0.43978538117018745, 0.43871052486774725, 0.4304687582595661, 0.4175887912767465, 0.3901967232596237, 0.42980754219207, 0.50220739093491885, 0.4122006868694329, 0.41101306091918375, 0.40266403742193705, 0.42602579315859225, 0.4353188674261685]
r_crispr_auprc_mismatch = [0.31549263036793995, 0.3740745878464649, 0.38355483598491064, 0.3852220325160804, 0.4026896528072899, 0.38920067492374516, 0.37565033992352803, 0.40218171431890226, 0.37475786726742183, 0.4112804934635258, 0.36461005809520913, 0.3473156201231854, 0.3779097514358848, 0.40329839392803185, 0.4241532502428708, 0.3603368687289499, 0.38725636841872296, 0.39219650172134357, 0.38783464661794737, 0.41230092633868776]
crispr_ip_auprc_mismatch = [0.17935841392365123, 0.12298198578563907, 0.12475044210400135, 0.14372766334739362, 0.16787370690295927, 0.1517032433558073, 0.12672437807648243, 0.12764335795251572, 0.24407479365112994, 0.19827038392907953, 0.10750160436394768, 0.13688279339606535, 0.1526540519977005, 0.16385126890548757, 0.11525612038173685, 0.22490957001739592, 0.19917585448558245, 0.2594921883032383, 0.11976467528262208, 0.16957003091948641]
crispr_net_auprc_mismatch = [0.42640740092218754, 0.3190569467826963, 0.3880444203473553, 0.4311005117880356, 0.3790875605479453, 0.3728412340785865, 0.3345446692562608, 0.4332806062365051, 0.3226598445346026, 0.36481551294018366, 0.38725695900586593, 0.2959588971748146, 0.41535143500516863, 0.4275492133224448, 0.34794067405892853, 0.3788237694732964, 0.3982675031554134, 0.2634639111861432, 0.3635638949936475, 0.3268301205708606]
deepcrispr_auprc_mismatch = [0.02278464463952097, 0.02053606755389617, 0.02223590339204374, 0.022239506628774323, 0.0218338628236853, 0.022260434004486147, 0.022030600915125875, 0.02109445019262771, 0.02070031563100381, 0.020982500606694574, 0.021286124195004977, 0.021002352966780057, 0.02119425094959664, 0.02445011821401616, 0.022924215957386456, 0.021480611415026097, 0.021107803597619483, 0.021808792856296673, 0.020845959074781168, 0.02291705545450847]
cnn_std_mismatch = [0.12232353145317822, 0.1264482924099477, 0.1648077805322489, 0.11719147312411121, 0.14299029097495755, 0.14590287544519193, 0.08250502935089866, 0.1622844714107885, 0.0946839979700762, 0.0823606706790279, 0.11961354212604594, 0.14619126648696912, 0.0888095034635768, 0.09261854120727428, 0.09446808654597667, 0.13473130848834594, 0.15768813731835804, 0.09144015167682318, 0.11576864220252867, 0.1943117508672938]
cfd_score_mismatch = [0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238, 0.3325722401377238]
y_fit = get_fitted_y(x_np, crispr_m_auprc_mismatch, degree)
plt.plot(x_np, y_fit, color='#ED1C24', label='CRISPR-M')
y_fit = get_fitted_y(x_np, r_crispr_auprc_mismatch, degree)
plt.plot(x_np, y_fit, color='#22B14C', label='R-CRISPR')
y_fit = get_fitted_y(x_np, crispr_ip_auprc_mismatch, degree)
plt.plot(x_np, y_fit, color='#00A2E8', label='CRISPR-IP')
y_fit = get_fitted_y(x_np, crispr_net_auprc_mismatch, degree)
plt.plot(x_np, y_fit, color='#FFC90E', label='CRISPR-NET')
y_fit = get_fitted_y(x_np, deepcrispr_auprc_mismatch, degree)
plt.plot(x_np, y_fit, color='#FFAEC9', label='DeepCRISPR')
y_fit = get_fitted_y(x_np, cnn_std_mismatch, degree)
plt.plot(x_np, y_fit, color='#A349A4', label='CNN_std')
y_fit = get_fitted_y(x_np, cfd_score_mismatch, degree)
plt.plot(x_np, y_fit, color='#B97A57', label='CFDScoring')

plt.xlabel('Random seed')
plt.ylabel('AUPRC')
plt.legend(loc="best")
plt.xlim(-5, 195)
plt.title('(b) Impact of Random Seed on AUPRC\non Mismatches-only Dataset')

def get_c_n_m(y, m, replace=False):
    auprc_10 = set()
    while len(auprc_10) < 100:
        auprc_10.add(tuple(np.random.choice(y, size=m, replace=replace)))
    return auprc_10

def get_avg(y):
    auprc_10 = list()
    for i in y:
        auprc_10.append(np.mean(i))
    return auprc_10

plt.rc("font", family="Times New Roman")
params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": 9,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
plt.rcParams.update(params)
# * fig(c)
print("ploting fig(c)...")
plt.subplot(4, 2, (3,4))
# crispr_m_auprc_5 = get_avg(get_c_n_m(crispr_m_auprc_circle_guide, 5))
# crispr_m_auprc_10 = get_avg(get_c_n_m(crispr_m_auprc_circle_guide, 10))
# crispr_m_auprc_15 = get_avg(get_c_n_m(crispr_m_auprc_circle_guide, 15))
# data = crispr_m_auprc_5 + crispr_m_auprc_10 + crispr_m_auprc_15
# categories = ['5']*100 + ['10']*100 + ['15']*100
# sns.violinplot(x=categories, y=data)
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
def draw_variance(data, color, label):
    stds = list()
    for i in range(2, 16):
        stds.append(np.std(get_avg(get_c_n_m(data, i)))**2)
    # # plt.plot(range(2, 16), stds)
    # slopes = np.diff(stds) / np.diff(range(2, 16))
    # # 归一化斜率
    # normalized_slopes = np.abs(slopes) / np.max(np.abs(slopes))
    # # 创建一个颜色映射
    # cmap = plt.get_cmap("YlOrRd")
    # colors = cmap(np.arange(cmap.N))
    # # 从黄色开始，即从颜色列表的一半开始
    # start = int(cmap.N / 4)
    # new_cmap = ListedColormap(colors[start:,:-1][::-1])
    # norm = mcolors.Normalize(vmin=0, vmax=1)
    # 绘制折线图
    # for i in range(len(slopes)):
    #     plt.plot(range(2+i, 4+i), stds[i:i+2], color=new_cmap(norm(normalized_slopes[i])))
    plt.plot(range(2, 16), stds, color=color, label=label)

print("ploting fig(c) - crispr_m_auprc_mismatch")
draw_variance(crispr_m_auprc_mismatch, color='#ED1C24', label='CRISPR-M')
print("ploting fig(c) - r_crispr_auprc_mismatch")
draw_variance(r_crispr_auprc_mismatch, color='#22B14C', label='R-CRISPR')
print("ploting fig(c) - crispr_ip_auprc_mismatch")
draw_variance(crispr_ip_auprc_mismatch, color='#00A2E8', label='CRISPR-IP')
print("ploting fig(c) - crispr_net_auprc_mismatch")
draw_variance(crispr_net_auprc_mismatch, color='#FFC90E', label='CRISPR-NET')
print("ploting fig(c) - deepcrispr_auprc_mismatch")
draw_variance(deepcrispr_auprc_mismatch, color='#FFAEC9', label='DeepCRISPR')
print("ploting fig(c) - cnn_std_mismatch")
draw_variance(cnn_std_mismatch, color='#A349A4', label='CNN_std')

plt.title('(c) Variance about the Number of Test Used for Averaging')
plt.legend(loc="best")
plt.xlabel('Number of Test Used for Averaging')
plt.ylabel('Variance')


# * fig(d)
print("ploting fig(d)...")
# 从(a)中的每个y值数组的20个元素中，随机不重复的抽取10个，计算平均值。抽取100次得到100个平均值，用每个y数组的100个平均值画小提琴图
plt.subplot(4, 2, 5)


crispr_m_auprc_10 = get_avg(get_c_n_m(crispr_m_auprc_circle_guide, 10))
r_crispr_auprc_10 = get_avg(get_c_n_m(r_crispr_auprc_circle_guide, 10))
crispr_ip_auprc_10 = get_avg(get_c_n_m(crispr_ip_auprc_circle_guide, 10))
crispr_net_auprc_10 = get_avg(get_c_n_m(crispr_net_auprc_circle_guide, 10))
# draw violin plot
data = crispr_m_auprc_10 + r_crispr_auprc_10 + crispr_ip_auprc_10 + crispr_net_auprc_10
categories = ['CRISPR-M']*100 + ['R-CRISPR']*100 + ['CRISPR-IP']*100 + ['CRISPR-NET']*100

# 使用seaborn的violinplot函数来画小提琴图
sns.violinplot(x=categories, y=data, scale='width')
plt.title('(d) Distribution of Average AUPRC from 10 Test\non CIRCLE_GUIDE Dataset')

# * fig(e)
print("ploting fig(e)...")
plt.subplot(4, 2, 6)
crispr_m_auprc_10 = get_avg(get_c_n_m(crispr_m_auprc_mismatch, 10))
r_crispr_auprc_10 = get_avg(get_c_n_m(r_crispr_auprc_mismatch, 10))
crispr_ip_auprc_10 = get_avg(get_c_n_m(crispr_ip_auprc_mismatch, 10))
crispr_net_auprc_10 = get_avg(get_c_n_m(crispr_net_auprc_mismatch, 10))
deepcrispr_auprc_10 = get_avg(get_c_n_m(deepcrispr_auprc_mismatch, 10))
cnn_std_auprc_10 = get_avg(get_c_n_m(cnn_std_mismatch, 10))
cfd_score_auprc_10 = [cfd_score_mismatch[0]]*100

data = crispr_m_auprc_10 + r_crispr_auprc_10 + crispr_ip_auprc_10 + crispr_net_auprc_10 + deepcrispr_auprc_10 + cnn_std_auprc_10 + cfd_score_auprc_10
categories = ['CRISPR-M']*100 + ['R-CRISPR']*100 + ['CRISPR-IP']*100 + ['CRISPR-NET']*100 + ['DeepCRISPR']*100 + ['CNN_std']*100 + ['CFDScoring']*100

sns.violinplot(x=categories, y=data, scale='width')
plt.xticks(rotation=30)
plt.title('(e) Distribution of Average AUPRC from 10 Test\non Mismatches-only Dataset')


path_list = ["./without_CNN/", "./without_LSTM/", "./without_Dense/", "./full_model/"]
path_list = ["../10ablation/"+i for i in path_list]
path2_list = ["./m81212_n13_without_branch12/", "./m81212_n13_without_branch34/", "./full_model/"]
path2_list = ["../10ablation/"+i for i in path2_list]
# model_name_list = ["without_CNN", "without_LSTM", "without_Dense", "full_model"]
model_name_list = ["Ablation model 1", "Ablation model 2", "Ablation model 3", "CRISPR-M"]
model_name_list2 = ["Ablation model a", "Ablation model b", "CRISPR-M"]
c = ["#FFC90E", "#22B14C", "#00A2E8", "#ED1C24", "#FFAEC9"]
FONTSIZE = 10
# plt.style.use("fast")
plt.rc("font", family="Times New Roman")
params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": 9,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
plt.rcParams.update(params)

plt.subplot(4, 2, 7)
for i in range(4):
    draw_mean_roc(path_list[i], c[i], model_name_list[i])

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('(f) Module Ablation: Mean Receiver Operating Characteristic Curve')
plt.legend(loc="best")

plt.subplot(4, 2, 8)
for i in range(4):
    draw_mean_prc(path_list[i], c[i], model_name_list[i])

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('(g) Module Ablation: Mean Precision-Recall Curve')
plt.legend(loc="best")

plt.tight_layout()
# plt.savefig(fname="random_seed.svg", format="svg", bbox_inches="tight")
# plt.savefig(fname="random_seed.tif", format="tif", bbox_inches="tight")
plt.savefig(fname="random_seed.png", format="png", bbox_inches="tight")
# plt.savefig(fname="random_seed.eps", format="eps", bbox_inches="tight")
# plt.show()
plt.close()
