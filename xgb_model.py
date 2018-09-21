clf = XGBClassifier()
booster = Booster()
booster.load_model('./model.xgb')
clf._Booster = booster