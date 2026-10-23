# The mock chain: how the data is generated

The generator (`src/demand_forecasting/data/`) builds a fictitious convenience-store chain and
simulates two things for it: what customers *wanted* every day, and what an ordinary ordering
system *managed to sell*. It is deterministic: the same preset and seed always give the same data.

## Sizes

| Preset | Countries | Stores | SKUs | Days of history | store x SKU x day rows | Where |
|---|---|---|---|---|---|---|
| `tiny` | 2 | 4 | 30 | 420 | ~40 thousand | tests and CI (seconds) |
| `small` | 3 | 24 | 120 | 730 | ~1.5 million | default local run (~2 min) |
| `medium` | 5 | 80 | 300 | 1,095 | ~25 million | laptop stress test, Free Edition upper end |
| `large` | 12 | 2,500 | 1,200 | 1,095 | ~3 billion | paid Databricks workspace only |

Not every store carries every SKU (about 85% of the assortment; fresh food 95%), and products
and stores come and go, so actual row counts are a little lower.

## The world (`world.py`)

* **Countries** (`domain.py`): 12 markets with their own climate (mean temperature, seasonal swing,
  rainy-day share, southern hemisphere flips the seasons), fixed-date public holidays and paydays.
  Larger markets get more stores.
* **Stores**: a format (urban 35%, suburban 25%, highway 15%, office 15%, transit 10%), a footfall
  index, a hidden growth trend, and about 10% open *during* the history (new-store cold start).
* **Products**: 27 categories in 8 aisles, from cold beverages and hot coffee to batteries and
  umbrellas. Each SKU gets a popularity, a list price, and about 15% launch during the history (new
  product cold start); about 6% are delisted.
* **Assortment and replenishment rules**: which store carries which SKU, the delivery lead time
  (fresh 1 day; ambient 2-3 days per store) and review period (fresh daily; ambient every 2, 3 or 7 days).
* **Calendar** per country: holidays and paydays, including the 14 future days.
* **Weather** per country: seasonal temperature plus multi-day warm/cold spells, rain, and a
  *forecast* issued the day before (observed weather + forecast error). The future has forecasts only.
* **Promotions**: about 45% of SKUs are promoted per country, about 5 campaigns a year of 7 or 14 days at
  10-30% off, planned into the future.
* **Prices**: about 30% of SKUs get one list-price increase (3-12%) during the history.

## What drives demand (`simulate.py`)

Expected demand for a store, SKU and day multiplies:

| Effect | Example |
|---|---|
| SKU popularity x store footfall | a popular soft drink in a busy store |
| Store format x category affinity | office stores sell 1.5x the coffee; highway stores 1.5x energy drinks |
| Weekday profile by format | office stores drop to 35-45% at weekends |
| Yearly season by category and hemisphere | ice cream peaks in summer (December in Australia) |
| Temperature (vs. the country's normal) | +6% per degree for ice cream, -2% for hot coffee |
| Rain | umbrellas x4 on rainy days, ice cream -20% |
| Holidays by format | highway +35%, office -60% |
| Paydays | small lift on discretionary items for 3 days |
| Promotions | lift = (1 - discount) ^ -elasticity, e.g. 20% off candy gives x1.56 |
| List price changes | mild drop after an increase |
| Store growth, launch ramp | new items and stores take about a month to reach normal |

**True demand** is drawn around that expectation from a negative-binomial distribution, which gives the
bursty, often-zero pattern of real convenience-store sales: close to 40% of store x SKU days sell nothing.

## The incumbent ordering system (the problem to solve)

Each store orders like a simple, common system does: on review days it orders up to
*smoothed recent sales x (review period + lead time) x a cover factor*. The cover factor varies by store and
SKU, from lean (frequent stockouts) to generous (overstock). It learns only from what it **sold**, with a
basic uplift after sold-out days, so it reacts late to heat waves and promotions. Fresh food is ordered daily
for the next day, and whatever is left at closing is written off.

On the `small` preset this incumbent serves about **91% of true demand**, and about 30% of fresh food is
written off. That is the baseline the forecast-driven policy is compared with.

## What the pipeline sees, and what it does not

| Table | Contains | Real-world analogue |
|---|---|---|
| `bronze.pos_sales` | units and revenue per store, SKU, day; **only days with sales** | point-of-sale system |
| `bronze.inventory_snapshots` | closing stock, deliveries received, write-offs | store inventory system |
| `bronze.purchase_orders` | order date, expected arrival, quantity | ordering system |
| `bronze.stores`, `products`, `assortment`, `calendar`, `weather`, `promotions`, `prices` | master and planning data | ERP, merchandising, weather feed |
| `bronze.simulation_truth` | true demand and expected demand | **does not exist in real life**; used only for evaluation |

The pipeline never reads `simulation_truth` to train or forecast, only to grade the backtest.
