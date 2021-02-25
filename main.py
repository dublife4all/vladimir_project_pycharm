# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot

code = 'С00-С96'
# code = ''

# plt = __import__("matplotlib.pyplot", globals(), locals(), [], 0)
# plt=__import__("matplotlib.pyplot")
# plt.pyplot.subplots
# import matplotlib.pyplot.subplots as plt
# import matplotlib.cm


def load_data():
    df = pd.read_excel('/Users/vladimir/Desktop/jupyter/in.xlsx', header=[0], nrows=212, skiprows=5, usecols='B:P',
                       index_col=False,
                       sheet_name='ф.12 т.2000, т.2001, т.2100')  # , usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    df.columns = ['Наименование классов и отдельных болезней', '№ строк', 'Код строки в Медстат',
                  'Код по МКБ-10 пересмотра', 'всего', 'из них: юнош', 'взято под диспансер- ное наблю- дение',
                  'с впервые в жизни установ- ленным диагнозом',
                  'из заболеваний с впервые в жизни установленным диагнозом (из гр. 9) взято под диспансер- ное наблю- дение',
                  'из заболеваний с впервые в жизни установленным диагнозом (из гр. 9) выявлено при проф- осмотре',
                  'из заболеваний с впервые в жизни установленным диагнозом (из гр. 9) выявлено при диспан-серизации',
                  'из заболе- ваний с впервые в жизни установ-ленном диагнозом (из гр.9) юноши',
                  'Снято с диспан- серного наблю- дения',
                  'Состоит под диспан- серным наблюде- нием на конец отчетного года', 'из них (из гр. 15): юноши']

    df.fillna(0, inplace=True)

    #     df = df[:212]

    df['из них: девушки'] = df['всего'] - df['из них: юнош']
    #     df['из них: девушки'] = df['из них: девушки'].astype(int)

    df['Наименование классов и отдельных болезней'] = df['Наименование классов и отдельных болезней'].replace('\n', '',
                                                                                                              regex=True)

    #     df['из них: девушки'] = df['из них: девушки'].astype(str)
    #     df['из них: девушки'] = df['из них: девушки'].replace('0', np.nan, regex=True)

    df.to_excel('out2.xlsx', sheet_name='Sheet1', index=False)

    #     (inplace  = True)

    #     for i in range(1, 212):
    # data = np.array(
    #     [[df['из них: девушки'][0], df['из них: юнош'][0]], [df['из них: девушки'][1], df['из них: юнош'][1]],
    #      [df['из них: девушки'][2], df['из них: юнош'][2]]])
    # #     data = data[0]
    # # for i in range(1, 3):
    # data = np.array([[df['из них: девушки'][0], df['из них: юнош'][0]], [df['из них: девушки'][1], df['из них: юнош'][1]]])
    # labels1 = [df['Код по МКБ-10 пересмотра'][0], df['Код по МКБ-10 пересмотра'][1]]
    #     # labels1 = ['По всем заболеваниям']
    # labels2 = ['м', 'ж', 'м', 'ж']
    # print(data)
    # print(labels1)
    # print(labels2)



    # data=np.array([])
    # data = [[]]
    # data = np.array[[], []]
    # data = []
    # data.append([])
    # data = [[], []]
    # data = np.empty((2, 2)).astype(int)
    # data = pd.DataFrame([[],[]])




    # col_names = ['A', 'B']
    # data = pd.DataFrame(columns=col_names)
    #
    # # arr[0, 1] = 'abc'
    # labels1=[]
    # labels2=[]
    # for i in range(1, 10):
    #     a = np.array([[df['из них: девушки'][i], df['из них: юнош'][i]]]).astype(int)
    #     # if a == np.array([0,0]):
    #     #     continue
    #     data = np.append(data, a, axis = 0)
    #     labels1.append(df['Код по МКБ-10 пересмотра'][i])
    #     labels2.append('м')
    #     labels2.append('ж')
    # print(data)
    # print(labels1)
    # print(labels2)


    #     return df

    # return data, labels1, labels2
    return df


data = load_data()


# df = load_data()
# get_data_by_code(load_data()[0], code)

def get_data_by_code(df, code):

    labels1 = []
    labels2 = []
    col_names = ['A', 'B']
    data = pd.DataFrame(columns=col_names)

    if code == '':

        # col_names = ['A', 'B']
        # data = pd.DataFrame(columns=col_names)

        # arr[0, 1] = 'abc'
        # labels1=[]
        # labels2=[]
        for i in range(1, 100):
            a = np.array([[df['из них: девушки'][i], df['из них: юнош'][i]]]).astype(int)
            # if a == np.array([0,0]):
            #     continue
            data = np.append(data, a, axis = 0)
            labels1.append(df['Код по МКБ-10 пересмотра'][i])
            labels2.append('м')
            labels2.append('ж')
        print(data)
        print(labels1)
        print(labels2)
        return data, labels1, labels2
    else:
        # i = df[df['Код по МКБ-10 пересмотра'] == code].index[0]
        # data = np.array([[df['из них: девушки'][i], df['из них: юнош'][i]]]).astype(int)
        # labels1 = [df['Код по МКБ-10 пересмотра'][i]]
        # labels2 = ['м', 'ж']
    # print(data)
    # print(labels1)
    # print(labels2)

        data = df.loc[df['Код по МКБ-10 пересмотра'] == code, ['из них: девушки', 'из них: юнош']]
        cols = ['девушки', 'юноши']
        data.columns = cols
        return data

        # a = np.array([[df['из них: девушки'][i], df['из них: юнош'][i]]]).astype(int)



    #     return df
    # return data, labels1, labels2
    # return i

get_data_by_code(data, code)

# def graphics1(data):
#     fig, ax = plt.pyplot.subplots()
#     offset = 0.4
#     #     data = np.array([[5, 10], [8, 15], [11, 9]])
#     cmap = plt.pyplot.get_cmap("tab20b")
#     b_colors = cmap(np.array([0, 8, 12]))
#     sm_colors = cmap(np.array([1, 2, 3, 9, 10, 11, 13, 14, 15]))
#     ax.pie(data.sum(axis=1), radius=1, colors=b_colors, wedgeprops=dict(width=offset, edgecolor='w'))
#     ax.pie(data.flatten(), radius=1 - offset, colors=sm_colors, wedgeprops=dict(width=offset, edgecolor='w'),
#            labels=['из них: юнош', 'из них: девушки'])
#
#
# #     ax.pie.labels="common X"
# #     ax.label("common Y")
#
# graphics1(load_data())


def graphics1(data, labels1, labels2):
    fig, ax = plt.pyplot.subplots()
    offset = 0.4
    #     data = np.array([[5, 10], [8, 15], [11, 9]])
    cmap = plt.pyplot.get_cmap("tab20b")
    b_colors = cmap(np.array([0, 8, 12]))
    sm_colors = cmap(np.array([1, 2, 3, 9, 10, 11, 13, 14, 15]))
    # labels1 = ['По всем заболеваниям']
    # labels2 = ['юноши', 'девушки', 'юноши', 'девушки']
    ax.pie(data.sum(axis=1), radius=1, colors=b_colors, wedgeprops=dict(width=offset, edgecolor='w'), labels=labels1)
    ax.pie(data.flatten(), radius=1 - offset, colors=sm_colors, wedgeprops=dict(width=offset, edgecolor='w'),
           labels=labels2, textprops=dict(color="w"))



    fig.show()


#     ax.pie.labels="common X"
#     ax.label("common Y")

graphics1(get_data_by_code()[0],get_data_by_code()[1],get_data_by_code()[2])

# graphics1(data,labels1,labels2)


fig, ax = plt.pyplot.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["375 g flour",
          "75 g sugar",
          "250 g butter",
          "300 g berries"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))



plt.pyplot.setp(autotexts, size=8, weight="bold")

ax.set_title("Matplotlib bakery: A pie")

# autotexts[0]=Text(0.1,0.3,'37.5%\n(375 gsss)')

plt.pyplot.show()

print(autotexts)




fig, ax = plt.pyplot.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["225 g flour",
          "90 g sugar",
          "1 egg",
          "60 g butter",
          "100 ml milk",
          "1/2 package of yeast"]

data = [225, 90, 50, 60, 100, 5]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Matplotlib bakery: A donut")

plt.pyplot.show()