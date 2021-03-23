# код болезни по мкб для выборки
code = 'С00-С96'
# code = ''

import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot

# загрузка данных из ексель документа
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
    df['из них: девушки'] = df['всего'] - df['из них: юнош']
    df['Наименование классов и отдельных болезней'] = df['Наименование классов и отдельных болезней'].replace('\n', '',
                                                                                                              regex=True)
    df.to_excel('out2.xlsx', sheet_name='Sheet1', index=False)
    return df

# выборка по коду мкб
def get_data_by_code_2(df, code):
    data = df.loc[
        df['Код по МКБ-10 пересмотра'] == code, ['Код по МКБ-10 пересмотра', 'из них: девушки', 'из них: юнош']]
    cols = ['Код по МКБ-10 пересмотра', 'девушки', 'юноши']
    data.columns = cols
    # print(np.array(data.columns))
    return data





# Построение графика

def graphics1_2(data):
    # i = df[df['Код по МКБ-10 пересмотра'] == code].index[0]
    # print(np.array([data['Код по МКБ-10 пересмотра']][0]))
    labels1 = np.array([data['Код по МКБ-10 пересмотра']])
    # labels2 = np.array(data.columns)
    labels2 = ['девушки', 'юноши']
    # data = np.array(data['девушки', 'юноши'])
    # data = data['девушки'], data['юноши']
    data = np.array([[data.iloc[0]['девушки'], data.iloc[0]['юноши']]])

    # mask = df['A'].values == 'foo'
    # data = np.array([[data['из них: девушки'], data['из них: юнош']]])
    print(data)
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




# функция исполнения
def run():
    df = load_data()
    data = get_data_by_code_2(df, code)
    graphics1_2(data)

run()