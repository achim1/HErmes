import pytest

from pyevsel.fitting import fit, model, functions

def funcfactory():
    def f(xs, p1, p2):
        return xs
    return f

def firstguessfactory():
    def f(xs):
        return [1,2]
    return f

def modelfactory():
    return model.Model(funcfactory(), startparams=(1,2))

def datafactory():
    import numpy as np

    return np.linspace(0,1000,10000)

def test_model_create_distribution():
    import numpy as np

    model = modelfactory()
    data = datafactory()
    model._create_distribution(data, 100, normalize=True)
    assert (isinstance(model.xs, np.ndarray) and isinstance(model.data, np.ndarray))

def test_model_add_first_guess():
    model = modelfactory()
    model.couple_all_models()
    model.add_first_guess(firstguessfactory())
    assert callable(model.first_guess)

def test_model_eval_first_guess():
    data = datafactory()
    model = modelfactory()
    first_guess = firstguessfactory()
    model.couple_all_models()
    model.add_first_guess(first_guess)
    model.eval_first_guess(data)
    assert model.startparams == first_guess(data)

def test_model_all_coupled():
    model = modelfactory()
    model.couple_all_models()
    assert model.all_coupled

def test_model_couple_models():
    model_a = modelfactory()
    model_b = modelfactory()
    model_c = model_a + model_b
    model_c.couple_models(0)
    assert model_c.coupling_variable == [0]

def test_model_add():
    model_a = modelfactory()
    model_b = modelfactory()
    assert isinstance(model_a + model_b, model.Model)

def test_model_components():
    model_a = modelfactory()
    model_b = modelfactory()
    model_c = model_a + model_b
    data = datafactory()
    models = [model_a, model_b]
    for i, result in enumerate([c(data) for c in model_c.components]):
        assert (result == models[i]._callbacks[0](data, *models[i].startparams)).all() 

    #assert [c(data) for c in model_c.components] == [model_a._callbacks[0](data, *model_a.startparams), model_b._callbacks[0](data, *model_b.startparams)] 

def test_model_call():
    model = modelfactory()
    data = datafactory()
    assert (model(data, *model.startparams) == data).all()

def test_model_call_mutliple():
    model = modelfactory() + modelfactory()
    data = datafactory()
    assert (model(data, *model.startparams) == 2*data).all()

def test_model_add_data():
    model = modelfactory()
    data = datafactory()
    model.add_data(data, xs = data)
    assert (model.data == data).all()

def test_model_add_data_distr():
    model = modelfactory()
    data = datafactory()
    model.add_data(data, create_distribution = True)
    assert (model.data == model._distribution.bincontent).all()

def test_model_fit_to_data():
    model = modelfactory()
    data = datafactory()
    model.add_data(data, xs = data)
    assert list(model.fit_to_data()) == list(model.best_fit_params)

def test_model_fit_to_data_distribution():
    model = modelfactory()
    data = datafactory()
    model.add_data(data,  create_distribution = True)
    assert list(model.fit_to_data()) == list(model.best_fit_params)

def test_model_plot_result():
    import pylab as p

    model = modelfactory()
    data = datafactory()
    model.add_data(data, xs = data)
    model.fit_to_data()
    assert isinstance(model.plot_result(), p.Figure)

def test_model_plot_result_distribution():
    import pylab as p

    model = modelfactory()
    data = datafactory()
    model.add_data(data,  create_distribution = True)
    model.fit_to_data()
    assert isinstance(model.plot_result(), p.Figure)

    
def test_model_clear():
    model = modelfactory()
    newmodel = model
    model.clear()
    assert model == newmodel

