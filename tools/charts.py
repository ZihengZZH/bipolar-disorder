import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


bi_MFCC_DM = [0.4166666666666667, 0.2814814814814815, 0.5277777777777778, 0.5873015873015873, 0.625, 0.5, 0.5317460317460317, 0.7936507936507936, 0.4444444444444444, 0.7083333333333334]
bi_MFCC_DBOW = [0.47222222222222215, 0.5333333333333333, 0.5833333333333334, 0.5238095238095238, 0.5, 0.6666666666666666, 0.5238095238095238, 0.8412698412698413, 0.27777777777777773, 0.5833333333333334]
bi_eGeMAPS_DM = [0.47222222222222215, 0.5444444444444444, 0.4166666666666667, 0.4841269841269842, 0.4166666666666667, 0.625, 0.35714285714285715, 0.626984126984127, 0.6666666666666666, 0.5138888888888888]
bi_eGeMAPS_DBOW = [0.4166666666666667, 0.6481481481481481, 0.2222222222222222, 0.5952380952380952, 0.375, 0.6666666666666666, 0.626984126984127, 0.626984126984127, 0.5277777777777778, 0.5416666666666666]
multi_MFCC_DM = [0.6388888888888888, 0.3481481481481481, 0.38888888888888884, 0.5793650793650794, 0.9166666666666666, 0.6666666666666666, 0.4761904761904762, 0.6904761904761906, 0.75, 0.5277777777777778]
multi_MFCC_DBOW = [0.5833333333333334, 0.45925925925925926, 0.5, 0.42063492063492064, 0.8333333333333334, 0.75, 0.42857142857142855, 0.746031746031746, 0.5833333333333334, 0.4166666666666667]
multi_eGeMAPS_DM = [0.38888888888888884, 0.4481481481481482, 0.6666666666666666, 0.5158730158730159, 0.5833333333333334, 0.5, 0.5952380952380952, 0.6825396825396824, 0.5555555555555556, 0.5]
multi_eGeMAPS_DBOW = [0.47222222222222215, 0.4851851851851852, 0.75, 0.626984126984127, 0.7083333333333334, 0.5416666666666666, 0.5317460317460317, 0.4761904761904761, 0.638888888888889, 0.5833333333333334]


def error_up(lst):
    return np.max(lst)-np.mean(lst)

def error_down(lst):
    return np.mean(lst)-np.min(lst)

def draw_bar_chart_cv():
    
    trace1 = go.Bar(
        x = [
            np.mean(bi_MFCC_DM),
            np.mean(bi_MFCC_DBOW),
            np.mean(bi_eGeMAPS_DM),
            np.mean(bi_eGeMAPS_DBOW),
            np.mean(multi_MFCC_DM),
            np.mean(multi_MFCC_DBOW),
            np.mean(multi_eGeMAPS_DM),
            np.mean(multi_eGeMAPS_DBOW)
        ][::-1],
        # y = [
        #     'bi_MFCC_DM',
        #     'bi_MFCC_DBOW',
        #     'bi_eGeMAPS_DM',
        #     'bi_eGeMAPS_DBOW',
        #     'multi_MFCC_DM',
        #     'multi_MFCC_DBOW',
        #     'multi_eGeMAPS_DM',
        #     'multi_eGeMAPS_DBOW',
        # ][::-1],
        y = [
            'fusion (1)',
            'fusion (2)',
            'fusion (3)',
            'fusion (4)',
            'fusion (5)',
            'fusion (6)',
            'fusion (7)',
            'fusion (8)',
        ][::-1],
        width = [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6],
        orientation='h',
        error_x=dict(
            type='data',
            color='rgb(240,131,91)',
            array=[
                error_up(bi_MFCC_DM),
                error_up(bi_MFCC_DBOW),
                error_up(bi_eGeMAPS_DM),
                error_up(bi_eGeMAPS_DBOW),
                error_up(multi_MFCC_DM),
                error_up(multi_MFCC_DBOW),
                error_up(multi_eGeMAPS_DM),
                error_up(multi_eGeMAPS_DBOW)
            ][::-1],
            arrayminus=[
                error_down(bi_MFCC_DM),
                error_down(bi_MFCC_DBOW),
                error_down(bi_eGeMAPS_DM),
                error_down(bi_eGeMAPS_DBOW),
                error_down(multi_MFCC_DM),
                error_down(multi_MFCC_DBOW),
                error_down(multi_eGeMAPS_DM),
                error_down(multi_eGeMAPS_DBOW)
            ][::-1],
            visible=True
        ),
        marker=dict(
            color=[
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue',
                'lightblue'
            ]
        ),
    )

    data = [trace1]
    layout=go.Layout(
        xaxis=dict(
            range=[0.2, 0.95],
            dtick=0.01
        ),
        font=dict(size=20),
        yaxis={'automargin': True}
    )
    
    plotly.offline.plot(go.Figure(data=data, layout=layout))

def draw_bar_chart():
    results = [
        0.357,
        0.392,
        0.399,
        0.452,
        0.291,
        0.307,
        0.429,
        0.384,
        0.373,
        0.492,
        0.426,
        0.402,
        0.429,
        0.439,
        0.373,
        0.415,
        0.452,
        0.294,
        0.373,
        0.378,
        0.370,
        0.505,
        0.439,
        0.399,
        0.463,
        0.437,
        0.315
    ]
    names = list(range(1,28))
    names = [str(x) for x in names]
    baseline = [0.489, 0.481]

    plt.bar(names[:18], results[:18], align='center', alpha=0.5, color='b')
    plt.bar(names[18:], results[18:], align='center', alpha=0.5, color='g')
    plt.hlines(baseline[0], -1, 27, label='baseline (BoAW)')
    plt.hlines(baseline[1], -1, 27, label='baseline (FAU)', color='r')
    plt.xlabel('Index')
    plt.ylabel('UAR')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print(np.mean(multi_MFCC_DM))
    print(np.mean(multi_MFCC_DBOW))
    print(np.mean(multi_eGeMAPS_DM))
    print(np.mean(multi_eGeMAPS_DBOW))
    
    draw_bar_chart_cv()