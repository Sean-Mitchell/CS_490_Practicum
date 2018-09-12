import sys
import pandas as pd

def main():
    for arg in sys.argv[1:]:
        df = pd.read_csv(arg, encoding="ISO-8859-1")
        
        
    #df = df.groupby(['cAmount', 'gammaAmount', 'Learning_Type'])    
    '''
    for name, group in df:
        print(name)
        print(group['F1_Score'].describe())
    '''
    pd.set_option('display.max_colwidth', -1)
    df2 = df.groupby(['cAmount', 'gammaAmount', 'Learning_Type'])['F1_Score'].describe()
    print(df2.head())
    #df2 = df.groupby(['cAmount', 'gammaAmount', 'Learning_Type']).describe().reset_index().pivot(index=['cAmount', 'gammaAmount', 'Learning_Type'], values='score',columns='level_1')
    #df2.to_csv('Scores.csv', encoding='utf-8')
main()