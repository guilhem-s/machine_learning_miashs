module GymnasticMedalsPrediction

# Importation des packages nécessaires
using Pkg
Pkg.add("RandomForest")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CategoricalArrays")
Pkg.add("MLJ")
Pkg.add("MLJModelInterface")
Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")
Pkg.add("MLJModels")
Pkg.add("MLJDecisionTreeInterface")
Pkg.add("MLJBase")

using MLJModels
using DecisionTree
using MLJ
using CSV
using DataFrames
using CategoricalArrays
using MLJBase: confusion_matrix, precision, recall, f1score

# Fonction pour préparer les données
data = CSV.read("Datasets/Athletes_summer_games.csv", DataFrame)

# Filtrer pour la gymnastique uniquement
databis = filter(row -> row[:Sport] in ["Artistic Gymnastics", "Gymnastics"], data)

# Filtrer pour les 5 derniers JO
filtered_data = filter(row -> row[:Year] >= 2008 && row[:Year] <= 2024, databis)
dropmissing!(filtered_data)
filtered_data.Medal = replace(filtered_data.Medal, missing => "No medal")
filtered_data.Medal = replace(filtered_data.Medal, "Gold" => 1, "Silver" => 2, "Bronze" => 3, "No medal" => 0)

# Nettoyage des événements
filtered_data.Event = categorical(filtered_data.Event)
replace!(filtered_data.Event, "Gymnastics Women's Uneven Bars" => "Barres asymétriques")
replace!(filtered_data.Event, "Gymnastics Women's Horse Vault" => "Saut Femme")
replace!(filtered_data.Event, "Gymnastics Women's Floor Exercise" => "Sol Femme")
replace!(filtered_data.Event, "Gymnastics Women's Balance Beam" => "Poutre Femme")
replace!(filtered_data.Event, "Gymnastics Men's Rings" => "Anneaux homme")
replace!(filtered_data.Event, "Gymnastics Men's Floor Exercise" => "Sol homme")
replace!(filtered_data.Event, "Gymnastics Men's Horse Vault" => "Saut homme")
replace!(filtered_data.Event, "Gymnastics Men's Parallel Bars" => "Barres parallèles hommes")
replace!(filtered_data.Event, "Gymnastics Men's Pommelled Horse" => "Cheval d'Arçons")
replace!(filtered_data.Event, "Gymnastics Men's Horizontal Bar" => "Barre fixe")
replace!(filtered_data.Event, "Gymnastics Women's Individual All-Around" => "Concours Individuel feminin")
replace!(filtered_data.Event, "Gymnastics Women's Team All-Around" => "Concours équipes feminin")
replace!(filtered_data.Event, "Gymnastics Men's Team All-Around" => "Concours équipes masculin")
replace!(filtered_data.Event, "Gymnastics Men's Individual All-Around" => "Concours Individuel masculin")

# Création de colonnes pour chaque type de médaille
filtered_data.GoldMedals = filtered_data.Medal .== 1
filtered_data.SilverMedals = filtered_data.Medal .== 2
filtered_data.BronzeMedals = filtered_data.Medal .== 3
filtered_data.NoMedals = filtered_data.Medal .== 0

# Transformation des variables en catégorielles
filtered_data.Sex = categorical(filtered_data.Sex)
filtered_data.NOC = categorical(filtered_data.NOC)
filtered_data.Name = categorical(filtered_data.Name)

filtered_data = dropmissing(filtered_data, :Age)
filtered_data.Age = Int.(filtered_data.Age)

# Nombre total de médailles par athlète
total_Medals = combine(groupby(filtered_data, :Name), [:GoldMedals, :SilverMedals, :BronzeMedals] .=> sum)
data = leftjoin(filtered_data, total_Medals, on = :Name)

    
# Sélection des caractéristiques et de la cible
features = select(data, Not(:Medal))
target = data.Medal

# Division des données en ensembles d'entraînement et de test
train, test = partition(eachindex(target), 0.7, shuffle=true)
X_train, X_test = features[train, :], features[test, :]
y_train, y_test = target[train], target[test]

# Charger le modèle RandomForestClassifier depuis DecisionTree
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree

# Configuration du modèle Random Forest
forest = RandomForestClassifier(n_trees=100, max_depth=5)

# Création de la machine avec les données d'entraînement
mach = machine(forest, X_train, y_train)

# Entraîner le modèle
fit!(mach)

# Création de la machine et entraînement
mach = machine(forest, X_train, y_train)
fit!(mach)
# Configurer le modèle
forest = RandomForestClassifier(n_trees=100, max_depth=5)

# Création de la machine en passant le modèle et les données d'entraînement
mach = machine(forest, X_train, y_train)

# Entraîner le modèle
fit!(mach)

# Prédiction sur les données de test
y_pred = predict(machine_model, X_test)

# Évaluation des performances du modèle
cm = confusion_matrix(y_test, y_pred)
println("Confusion Matrix: ", cm)

precision_score = precision(cm)
recall_score = recall(cm)
f1 = f1score(cm)

println("Precision: ", precision_score)
println("Recall: ", recall_score)
println("F1 Score: ", f1)



end # Fin du module GymnasticMedalsPrediction
