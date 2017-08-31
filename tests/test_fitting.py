import pytest
import numpy as np

from HErmes.fitting import fit, model, functions

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

def test_model_extract_paramters():
    model = modelfactory()
    model.couple_models(0)
    assert len(model.extract_parameters()) ==  2
    assert model.extract_parameters()[1] == [0]

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

def test_model_call_multiple():
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

def test_gauss():
    assert functions.gauss(1, 0, .2) == functions.n_gauss(1, 0, .2, 1)
    assert isinstance(functions.calculate_sigma_from_amp(1), float)
    data = np.random.normal(0, .2, 10000)
    gaussmod = model.Model(functions.gauss)

    gaussmod.startparams = [-0.2, .5]
    gaussmod.add_data(data, create_distribution=True,\
                      normalize=True, density=True)
    assert gaussmod.ndf == 198

    gaussmod.fit_to_data(errordef=1)
    fig = gaussmod.plot_result(xmax=5)
    fig.savefig("ptestgauss.png")
    assert 0.5 < gaussmod.chi2_ndf < 1.2
    assert -0.1 < gaussmod.best_fit_params[0] < .1
    assert .18 < gaussmod.best_fit_params[1] < .22

def test_poisson():

    data = np.random.poisson(100, size=10000)

    mod = model.Model(functions.poisson)

    mod.startparams = [80]
    mod.add_data(data, bins=200,\
                 create_distribution=True,\
                 normalize=True,\
                 density=False)
    assert mod.ndf == 199

    mod.fit_to_data(errordef=1)
    fig = mod.plot_result(xmax=150)
    #fig.savefig("ptest.png")
    assert 90 < mod.best_fit_params[0] < 110
    assert 0.5 < mod.chi2_ndf < 1.2

def test_chi2():

    assert functions.calculate_chi_square(np.array([1]), np.array([1])) == 0
    chi2 = functions.calculate_chi_square(np.random.normal(0, .2, 1000), np.random.normal(0, .2, 1000))
    assert isinstance(chi2, float)

def test_exponential():
    assert functions.exponential(0, 5) == 1

    mod = model.Model(lambda x, y: (1. / y) * functions.exponential(x, (1. / y)))
    beta = 20.

    data = np.random.exponential(beta, size=10000)
    mod.startparams = [19.]
    mod.add_data(data, create_distribution=True, normalize=True, density=False)

    assert mod.ndf == 199

    mod.fit_to_data(limits=((1,100),))
    assert 0.1 < mod.chi2_ndf < 20
    assert 15 < mod.best_fit_params[0] < 25 #FIXME: fit is really bad...

def test_pandel():
    pd = functions.pandel_factory(250000)
    assert isinstance(pd(np.linspace(0,2500,100), 50),np.ndarray)

#def test_fitmodel()
#    pass
