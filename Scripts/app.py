from flask import Flask, render_template, request, Response
import pandas as pd
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.multivariate.manova import MANOVA
from testset_prediction import testset_prediction as tsp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'amit'

lda=LinearDiscriminantAnalysis()

def corrl(df):
    lt=[]
    df1=df.iloc[:,0:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    return max(flat_list),min(flat_list)

def fit_linear_reg(X,y):
        dp=pd.concat([X,y],axis=1)
        table=MANOVA.from_formula('X.values~ y.values', data=dp).mv_test().results['y.values']['stat']
        Wilks_lambda=table.iloc[0,0]
        F_value=table.iloc[0,3]
        p_value=table.iloc[0,4]
        return Wilks_lambda,F_value,p_value,table

def writefile2(X,y,model):
    ts=tsp(X,y,model)
    tp,tn,fp,fn,sn,sp,acc,f1,mcc,roc=ts.fit()
    return tp,tn,fp,fn,sn,sp,acc,f1,mcc,roc
    '''
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    filerw.write('MCC: '+str(a9)+"\n")
    filerw.write('ROC_AUC: '+str(a10)+"\n")
    '''

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET','POST'])
def data():
    if request.method == 'POST':
       file_tr = request.form['csvfile_tr']
       data_tr = pd.read_csv(file_tr)
       file_ts = request.form['csvfile_ts']
       data_ts = pd.read_csv(file_ts)
       ntr=data_tr.iloc[:,0:1]
       #nts=data_ts.iloc[:,0:1]
       if request.form['options']=='first':
          Xtr=data_tr.iloc[:,2:]
          ytr=data_tr.iloc[:,1:2]
       elif request.form['options']=='last':
          Xtr=data_tr.iloc[:,1:-1]
          ytr=data_tr.iloc[:,-1:]
       #calculation of intercorrelation
       global dc
       dc=Xtr.corr()
       mx,mn=corrl(dc)
       mc=max(mx,mn)
       #formation of the equation
       lda.fit(Xtr,ytr)
       ic=lda.intercept_
       cf=list(lda.coef_[0])
       xcol=list(Xtr.columns)
       ls=[round(ic[0],3)]
       equation = ' + '.join([str(round(i,3))+"*{}".format(j) for i,j in zip(cf,xcol)])+' + '+str(round(ic[0],3))
       #goodness of fit
       L,F,P,T=fit_linear_reg(Xtr,ytr)
       #Sub-train results table
       tp1,tn1,fp1,fn1,sn1,sp1,acc1,f1,mcc1,roc1=writefile2(Xtr,ytr,lda)
       #Screening set testset_prediction
       Xts=data_ts[Xtr.columns]
       ytspr=pd.DataFrame(lda.predict(Xts), columns=['Pred'])
       ytspr2=pd.DataFrame(lda.predict_proba(Xts), columns=['%Prob(-1)','%Prob(1)'])
       global dfts
       dfts=pd.concat([data_ts,ytspr, ytspr2], axis=1)
       return render_template('data.html', trsize=ntr.shape[0],
              mc=mc,equation=equation,L=round(L,3), F=round(F,3), P=P,
              tp1=tp1, tn1=tn1, fp1=fp1, fn1=fn1, acc1=round(acc1,3), mcc1=round(mcc1,3)
              )

@app.route('/resultsTS')
def results_ts():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =dfts.to_html(index=False))

@app.route('/correlmatrix')
def correlmatrix():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =dc.to_html())


if __name__=='__main__':
    app.run(debug=True)
