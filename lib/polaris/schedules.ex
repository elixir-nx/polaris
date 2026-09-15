defmodule Polaris.Schedules do
  @moduledoc """
  Parameter Schedules.

  Parameter schedules are often used to anneal hyperparameters
  such as the learning rate during the training process. Schedules
  provide a mapping from the current time step to a learning rate
  or another hyperparameter.

  Choosing a good learning rate and consequently a good learning
  rate schedule is typically a process of trial and error. Learning
  rates should be relatively small such that the learning curve
  does not oscillate violently during the training process, but
  not so small that learning proceeds too slowly. Using a
  schedule slowly decreases oscillations during the training
  process such that, as the model converges, training also
  becomes more stable.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn

  @doc """
  Linear decay schedule.

  ## Options

    * `:warmup` - scheduler warmup steps. Defaults to `0`

    * `:steps` - total number of decay steps. Defaults to `1000`
  """
  def linear_decay(init_value, opts \\ []) do
    &apply_linear_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_linear_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        warmup: 0,
        steps: 1000
      )

    scale =
      if step < opts[:warmup] do
        step / Nx.max(1, opts[:warmup])
      else
        Nx.max(0.0, (opts[:steps] - step) / Nx.max(1, opts[:steps] - opts[:warmup]))
      end

    scale * opts[:init_value]
  end

  @doc ~S"""
  Exponential decay schedule.

  $$\gamma(t) = \gamma_0 * r^{\frac{t}{k}}$$

  ## Options

    * `:decay_rate` - rate of decay. $r$ in above formulation.
      Defaults to `0.95`

    * `:transition_steps` - steps per transition. $k$ in above
      formulation. Defaults to `10`

    * `:transition_begin` - step to begin transition. Defaults to `0`

    * `:staircase` - discretize outputs. Defaults to `false`

  """
  def exponential_decay(init_value, opts \\ []) do
    &apply_exponential_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_exponential_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        decay_rate: 0.95,
        transition_steps: 10,
        transition_begin: 0,
        staircase: false
      )

    init_value = opts[:init_value]
    rate = opts[:decay_rate]
    staircase? = opts[:staircase]
    k = opts[:transition_steps]
    start = opts[:transition_begin]

    t = Nx.subtract(step, start)

    p =
      if staircase? do
        t
        |> Nx.divide(k)
        |> Nx.floor()
      else
        t
        |> Nx.divide(k)
      end

    decayed_value =
      rate
      |> Nx.pow(p)
      |> Nx.multiply(init_value)

    Nx.select(
      Nx.less_equal(t, 0),
      init_value,
      decayed_value
    )
  end

  @doc ~S"""
  Cosine decay schedule.

  $$\gamma(t) = \gamma_0 * \left(\frac{1}{2}(1 - \alpha)(1 + \cos\pi \frac{t}{k}) + \alpha\right)$$

  ## Options

    * `:decay_steps` - number of steps to apply decay for.
      $k$ in above formulation. Defaults to `10`

    * `:alpha` - minimum value of multiplier adjusting learning rate.
      $\alpha$ in above formulation. Defaults to `0.0`

  ## References

    * [SGDR: Stochastic Gradient Descent with Warm Restarts](https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx)

  """
  def cosine_decay(init_value, opts \\ []) do
    &apply_cosine_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_cosine_decay(step, opts \\ []) do
    opts = keyword!(opts, init_value: 1.0e-2, decay_steps: 10, alpha: 0.0)
    init_value = opts[:init_value]
    decay_steps = opts[:decay_steps]
    alpha = opts[:alpha]

    theta = Nx.min(step, decay_steps) / decay_steps * Nx.Constants.pi()

    cos = (Nx.cos(theta) + 1) / 2 * (1 - alpha)

    init_value * (cos + alpha)
  end

  @doc ~S"""
  One-cycle schedule.

  Anneals from an initial value up to `peak_value` over the first
  `pct_start` fraction of `total_steps`, then back down to a minimum
  for the rest of training:

  $$\gamma_{init} = rac{\gamma_{peak}}{d}, \quad \gamma_{min} = rac{\gamma_{init}}{d_{final}}$$

  Each phase interpolates between its start and end value, with a
  cosine by default or linearly with `anneal: :linear`. With
  `three_phase: true` the middle phase anneals back to the initial
  value first and a third phase anneals from there to the minimum,
  which is how NeuralProphet and other Lightning based libraries
  run it.

  Matches `torch.optim.lr_scheduler.OneCycleLR`, including its defaults
  and phase boundaries, so learning rate recipes ported from PyTorch
  behave the same.

  ## Options

    * `:total_steps` - total number of steps in the cycle. Required

    * `:pct_start` - fraction of the cycle spent increasing the value.
      Defaults to `0.3`

    * `:anneal` - `:cos` or `:linear`. Defaults to `:cos`

    * `:div_factor` - $d$ in above formulation, `peak_value / initial_value`.
      Defaults to `25.0`

    * `:final_div_factor` - $d_{final}$ in above formulation,
      `initial_value / min_value`. Defaults to `1.0e4`

    * `:three_phase` - anneal back to the initial value before decaying
      to the minimum. Defaults to `false`

  ## References

    * [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)

  """
  def one_cycle(peak_value, opts \\ []) do
    total_steps = Keyword.fetch!(opts, :total_steps)
    pct_start = Keyword.get(opts, :pct_start, 0.3)
    div_factor = Keyword.get(opts, :div_factor, 25.0)
    final_div_factor = Keyword.get(opts, :final_div_factor, 1.0e4)
    three_phase = Keyword.get(opts, :three_phase, false)
    anneal = Keyword.get(opts, :anneal, :cos)

    initial_value = peak_value / div_factor
    min_value = initial_value / final_div_factor

    # Phase boundaries follow PyTorch's OneCycleLR exactly
    phase_1_end = pct_start * total_steps - 1
    phase_2_end = if three_phase, do: 2 * pct_start * total_steps - 2, else: total_steps - 1
    phase_2_target = if three_phase, do: initial_value, else: min_value

    &apply_one_cycle(&1,
      peak_value: peak_value,
      initial_value: initial_value,
      min_value: min_value,
      phase_1_end: phase_1_end,
      phase_2_end: phase_2_end,
      phase_2_target: phase_2_target,
      total_end: total_steps - 1,
      three_phase: three_phase,
      anneal: anneal
    )
  end

  defnp apply_one_cycle(step, opts \\ []) do
    opts =
      keyword!(opts, [
        :peak_value,
        :initial_value,
        :min_value,
        :phase_1_end,
        :phase_2_end,
        :phase_2_target,
        :total_end,
        :three_phase,
        :anneal
      ])

    step = Nx.as_type(step, :f32)

    phase_1 =
      anneal(opts[:initial_value], opts[:peak_value], step / opts[:phase_1_end], opts[:anneal])

    phase_2 =
      anneal(
        opts[:peak_value],
        opts[:phase_2_target],
        (step - opts[:phase_1_end]) / (opts[:phase_2_end] - opts[:phase_1_end]),
        opts[:anneal]
      )

    phase_3 =
      anneal(
        opts[:initial_value],
        opts[:min_value],
        (step - opts[:phase_2_end]) / (opts[:total_end] - opts[:phase_2_end]),
        opts[:anneal]
      )

    if opts[:three_phase] do
      Nx.select(
        step <= opts[:phase_1_end],
        phase_1,
        Nx.select(step <= opts[:phase_2_end], phase_2, phase_3)
      )
    else
      Nx.select(step <= opts[:phase_1_end], phase_1, phase_2)
    end
  end

  deftransformp anneal(start, finish, pct, anneal) do
    case anneal do
      :cos -> cosine_anneal(start, finish, pct)
      :linear -> linear_anneal(start, finish, pct)
    end
  end

  defnp cosine_anneal(start, finish, pct) do
    pct = Nx.clip(pct, 0.0, 1.0)
    finish + (start - finish) / 2 * (Nx.cos(Nx.Constants.pi() * pct) + 1)
  end

  defnp linear_anneal(start, finish, pct) do
    pct = Nx.clip(pct, 0.0, 1.0)
    (finish - start) * pct + start
  end

  @doc ~S"""
  Constant schedule.

  $$\gamma(t) = \gamma_0$$

  """
  def constant(init_value, opts \\ []) do
    &apply_constant(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_constant(_step, opts \\ []) do
    opts = keyword!(opts, init_value: 0.01)
    opts[:init_value]
  end

  @doc ~S"""
  Polynomial schedule.

  $$\gamma(t) = (\gamma_0 - \gamma_n) * (1 - \frac{t}{k})^p$$

  ## Options

    * `:end_value` - end value of annealed scalar. $\gamma_n$ in above formulation.
      Defaults to `1.0e-3`

    * `:power` - power of polynomial. $p$ in above formulation. Defaults to `2`

    * `:transition_steps` - number of steps over which annealing takes place.
      $k$ in above formulation. Defaults to `10`

  """
  def polynomial_decay(init_value, opts \\ []) do
    &apply_polynomial_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_polynomial_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        end_value: 1.0e-3,
        power: 2,
        transition_steps: 10,
        transition_begin: 0
      )

    init_value = opts[:init_value]
    end_value = opts[:end_value]
    start = opts[:transition_begin]
    k = opts[:transition_steps]
    p = opts[:power]

    step
    |> Nx.subtract(start)
    |> Nx.clip(0, k)
    |> Nx.divide(k)
    |> Nx.negate()
    |> Nx.add(1)
    |> Nx.pow(p)
    |> Nx.multiply(Nx.subtract(init_value, end_value))
    |> Nx.add(end_value)
  end
end
