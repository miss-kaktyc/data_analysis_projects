SELECT Products.name, Categories.name
FROM Products LEFT JOIN Categories ON Products.id=Categories.product_id;