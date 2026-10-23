# What this project does, in plain English

## The problem
A convenience store carries a few thousand products in a small space. Every few days someone
decides how many of each to order. Order too few and the shelf is empty when a customer comes
in: the sale is lost, and often the customer goes elsewhere next time. Order too many and money sits
on the shelf, and fresh food (sandwiches, hot food, bakery) goes in the bin at closing time.

Multiply that by thousands of stores in a dozen countries, every day, and the ordering
decision is one of the biggest levers on a chain's profit.

## What we built
A prototype "forecast-to-order" system:

1. **It learns from history.** For every store and product it looks at past daily sales, along with
   what drives them: the day of the week, holidays and paydays, promotions and prices, the
   weather forecast (ice cream sells in the heat, umbrellas in the rain), the kind of store
   (an office-district store is empty at weekends; a highway store is busy on holidays) and how
   long the product has been on sale.
2. **It forecasts the next 14 days**, one number per store, product and day. It also gives a
   "high but plausible" number: demand should stay below it on 9 days out of 10.
3. **It turns forecasts into orders.** For each store and product it works out how much stock
   is needed until the next delivery arrives, adds a safety buffer sized by how uncertain the
   forecast is, subtracts what is already on the shelf or on its way, and recommends an order quantity.
4. **It proves itself on the past first.** Before trusting it, we pretend to be at three
   earlier dates, let it forecast using only what was known then, and compare with what really happened,
   against two simple rules a store might use today.

## The trap we had to handle
When the shelf is empty, the sales figure says "3 sold", but customers wanted more. A naive
model learns "this product sells 3", orders 3, the shelf empties again, and the problem
reinforces itself. We detect days that ended with an empty shelf and estimate what demand
probably was on those days. Because the data is simulated, we also know the true demand, and we
measured that ignoring this trap would have made forecasts **15% too low** overall (35% for fresh food).

## How well does it work?
On the default mock chain (24 stores, 120 products, 2 years), compared with ordering based on a
28-day sales average (the full numbers are in [modeling.md](modeling.md#results)):

* forecasts are closer to true demand, and much less biased,
* the order recommendations would have avoided a stockout in **95% of ordering cycles** (the target),
  where the moving-average rule managed under half,
* **99% of demand would have been served**, against 82% with the moving-average rule and 91% for
  the store's current (simulated) ordering system.

The price of that service is more stock on the shelf. The safety level is a single setting
that the business can tune per category once the costs of waste and lost sales are known.

## What it is not (yet)
The data is **synthetic**: a realistic simulation, not a real chain. The numbers show that the
method and the pipeline work, not what a specific retailer would gain. It runs on a laptop,
and the same code has run on Databricks Free Edition with the same results
(see [databricks-migration.md](databricks-migration.md)).

## Where to go next
* Run it: the [README](../README.md) quickstart.
* How the pieces fit: [architecture.md](architecture.md).
* How the fake chain is generated: [mock-data.md](mock-data.md).
* The modelling, metrics and results: [modeling.md](modeling.md).
* Why each design choice was made: [decisions/](decisions/README.md).
* Words you might not know: [glossary.md](glossary.md).
