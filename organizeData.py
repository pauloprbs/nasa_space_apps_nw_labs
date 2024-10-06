import pandas as pd
import numpy as np
import os
import dateutil


is_None = lambda _,x : True if x is None else False
re_order_date = lambda _,date:  f"{date.split('/')[-1]}-{date.split('/')[-2]}-{date.split('/')[-3]}" if isinstance(date,str) else date
normalizeStr = lambda x: str(x).strip().lower().translate(str().maketrans("áàãâéèêíìîóòõôúùûñç","aaaaeeeiiioooouuunc"))
formatDates = lambda x: dateutil.parser.parse(x).strftime("%Y-%m-%dT%H:%M:%S.%f") if isinstance(x,str) else x

def timeTreat(date_string,date_format="dmy"):
        date_string = str(date_string).replace("/","").replace("-","").replace("\\","")
        # print(date_string)
        if date_format == "dmy":
            day = date_string[:2]
            month = date_string[2:4]
            year = date_string[4:]

        elif date_format == "mdy":
            month = date_string[:2]
            day = date_string[2:4]
            year = date_string[4:]

        elif date_format == "ymd":
            year = date_string[:4]
            month = date_string[4:6]
            day = date_string[6:]
            
        formatted_date = '-'.join([year,month])
        return formatted_date

def processTime(self,df=None,timeCols=None):
       
        if self.is_None(df) and self.is_None(timeCols):
            df,timeCols = self.df,self.timeCols
        
        DE,ATE = f"{timeCols}_De",f"{timeCols}_Ate"
        def process(time):
            if df.loc[1,time].find("a")!=-1:
                df[DE],df[ATE] = df[time],df[time]
                for i in df.index:
                    date = normalizeStr(df.loc[i,time])
                    if isinstance(date,str):
                        dates = date.replace(" ","").split("a")
                        # print(dates[0],dates[1])
                        date0,date1 = self.formatDates(dates[0]),self.formatDates(dates[1])
                        # print(date0,date1)
                        df.loc[i,DE],df.loc[i,ATE] = self.re_order_date(date0),self.re_order_date(date1)
                    else: continue
                df.drop(time,axis=1,inplace=True)
                self.timeCols = (DE,ATE)
            else:
                df[time] = df[time].apply(normalizeStr) 
                df[time] = df[time].apply(self.formatDates) 
                df[time] = df[time].apply(self.re_order_date)
