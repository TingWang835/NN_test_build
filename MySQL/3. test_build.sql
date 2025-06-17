-- Create Tables, import/enter data
-- <<drug_info>>
CREATE TABLE drug_info (
    drugID INT primary key AUTO_INCREMENT, 
    launch_date DATE,
    chemical_name VARCHAR(255),  
    brand_name VARCHAR(255),
    dosage_form VARCHAR(255),
    dosage VARCHAR(255), 
    indication TEXT,
    project_manager INT
);

SET GLOBAL LOCAL_INFILE=TRUE;

LOAD DATA LOCAL INFILE 'D:/Work repository/NN_test_build/MySQL/full_drug_list.csv'
    INTO TABLE drug_info
    FIELDS TERMINATED BY ','  
    ENCLOSED BY '"' 
    LINES TERMINATED BY '\n' 
    IGNORE 1 LINES;
    
    
-- <<sales>>
drop table sales;
CREATE TABLE sales (
    drugID INT primary key AUTO_INCREMENT, 
    price DECIMAL(10,2) DEFAULT 0,
    sold INT DEFAULT 0,
    total_sales DECIMAL(30,2),
    cost DECIMAL(10,2),
    total_costs DECIMAL(30,2)
    );

INSERT INTO sales (drugid, price, sold, cost)
VALUES (1, 25.6, 58620, 15.9),
       (2, 63.7, 106546, 20.4),
       (3, 66.5, 156283, 10.7),
       (4, 536.8, 185321, 100.2),
       (5, 85.9, 175269, 19.8),
       (6, 85.2, 16348, 22.5),
       (7, 718.2, 124563, 98.9),
       (8, 125.7, 175634, 20.1),
       (9, 17.2, 556934, 3.8),
       (10, 12.3, 446982, 2.7);

-- <<empolyees>>
CREATE TABLE empolyees (
    empolyee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    job VARCHAR(255),
    hourly_pay DECIMAL(10,2),
    working_hours DECIMAL(10,2) DEFAULT 2080,
    salary DECIMAL(15,2),
    hire_date DATE,
    superviser INT
);
ALTER TABLE empolyees AUTO_INCREMENT = 1000;

INSERT INTO empolyees (empolyee_id, first_name, last_name, job, hourly_pay, working_hours, hire_date, superviser)
VALUES (1000, "Adam", "Smith", "Department Manager", 100, 2080, "2013-01-08", null),
       (1001,"David", "Finch", "Chief Researcher", 80, 2080, "2013-01-09", 1000), 
       (1002, "Samatha", "Cody", "Researcher I", 65, 2080, "2013-02-06", 1001), 
       (1003, "John", "Smith", "Researcher I", 60, 2080, "2016-06-09", 1001), 
       (1004, "Mary", "James", "Researcher II", 45, 2079.0, "2017-08-22", 1002),
       (1005, "Jon", "Dean", "Researcher III", 30, 2084.3, "2018-04-30", 1004),
       (1006, "Janice", "Alice", "Researcher II", 40, 2080, "2019-05-18", 1003),
       (1007, "Jacob", "Smith", "Researcher III", 25, 1040.0, "2020-11-10", 1006);
	
  
-- <<expense & income>>
create table expense (
     expense_id INT primary key AUTO_INCREMENT,
     expense_type VARCHAR(255),
     amount DECIMAL(30,2)
);

INSERT into expense (expense_type)
values ("salaies"), ("production cost"), ("sales tax");

create table income (
     income_id INT primary key AUTO_INCREMENT,
     income_type VARCHAR(255),
     amount DECIMAL(30,2)
);

INSERT into income (income_type)
values ("product sales");

-- Foreign keys
ALTER TABLE sales
ADD CONSTRAINT fk_drug_info_id
FOREIGN KEY (DrugID) REFERENCES drug_info(DrugID);

ALTER TABLE drug_info
ADD CONSTRAINT fk_empolyees_pro_manager
FOREIGN KEY (project_manager) REFERENCES empolyees(empolyee_id)
on delete set null;


-- Triggers (naming rule "affected column/row_ref table_linked table_before/after_insert/update/delete")
-- triggers for <<sales>>
CREATE TRIGGER total_sale_sales_bf_ins
BEFORE INSERT ON sales
FOR EACH ROW
SET NEW.total_sales = NEW.price * NEW.sold;

CREATE TRIGGER total_sale_sales_bf_upd
BEFORE UPDATE ON sales
FOR EACH ROW
SET NEW.total_sales = NEW.price * NEW.sold;

CREATE TRIGGER total_costs_sales_bf_ins
BEFORE INSERT ON sales
FOR EACH ROW
SET NEW.total_costs = NEW.cost * NEW.sold;

CREATE TRIGGER total_costs_sales_bf_upd
BEFORE UPDATE ON sales
FOR EACH ROW
SET NEW.total_costs = NEW.cost * NEW.sold;

-- triggers for <<empolyees>>
CREATE TRIGGER salary_empolyees_bf_ins
BEFORE INSERT ON empolyees
FOR EACH ROW
SET NEW.salary = NEW.hourly_pay * NEW.working_hours;

CREATE TRIGGER salary_empolyees_bf_upd
BEFORE UPDATE ON empolyees
FOR EACH ROW
SET NEW.salary = NEW.hourly_pay * NEW.working_hours;

-- triggers between <<expense>> and <<empolyees>>
CREATE TRIGGER salaryies_empolyees_expense_af_ins
AFTER INSERT ON empolyees
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(salary) FROM empolyees)
WHERE expense_id = 1;

CREATE TRIGGER salaryies_empolyees_expense_af_upd
AFTER UPDATE ON empolyees
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(salary) FROM empolyees)
WHERE expense_id = 1;

CREATE TRIGGER salaryies_empolyees_expense_af_del
AFTER DELETE ON empolyees
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(salary) FROM empolyees)
WHERE expense_id = 1;

-- triggers between <<expense>> and <<sales>>
CREATE TRIGGER production_cost_sales_expense_af_ins
AFTER INSERT ON sales
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(total_costs) FROM sales)
WHERE expense_id = 2;

CREATE TRIGGER production_cost_sales_expense_af_upd
AFTER UPDATE ON sales
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(total_costs) FROM sales)
WHERE expense_id = 2;

CREATE TRIGGER production_cost_sales_expense_af_del
AFTER DELETE ON sales
FOR EACH ROW
UPDATE expense
SET amount = (SELECT SUM(total_costs) FROM sales)
WHERE expense_id = 2;

-- triggers between <<income>> and <<sales>>
CREATE TRIGGER product_sales_sales_expense_af_ins
AFTER INSERT ON sales
FOR EACH ROW
UPDATE income
SET amount = (SELECT SUM(total_sales) FROM sales)
WHERE income_id = 1;

CREATE TRIGGER product_sales_sales_expense_af_upd
AFTER UPDATE ON sales
FOR EACH ROW
UPDATE income
SET amount = (SELECT SUM(total_sales) FROM sales)
WHERE income_id = 1;

CREATE TRIGGER product_sales_sales_expense_af_del
AFTER DELETE ON sales
FOR EACH ROW
UPDATE income
SET amount = (SELECT SUM(total_sales) FROM sales)
WHERE income_id = 1;

-- Triggers between <<income>> and <<expense>>
CREATE TRIGGER sales_tax_income_expense_af_upd
AFTER UPDATE ON income
FOR EACH ROW
UPDATE expense
SET amount = (SELECT amount FROM income WHERE income_id = 1) * 0.06
WHERE expense_id = 3;

-- Views
CREATE VIEW product_sales AS
SELECT a.drugid, 
concat(a.chemical_name, " ", "(", a.brand_name, ")") AS "product_name", 
a.dosage, b.price, b.sold, b.total_sales
FROM drug_info AS a
LEFT JOIN sales AS b
ON a.drugid = b.drugid;

SELECT * FROM product_sales;

CREATE VIEW manage_project AS
SELECT a.empolyee_id,
      concat(a.first_name, " ", a.last_name) AS "empolyee_name",
      b.chemical_name, b.brand_name, dosage
FROM empolyees AS a
LEFT JOIN drug_info AS b
ON a.empolyee_id = b.project_manager;

select * from manage_project;

CREATE VIEW project_count AS
SELECT empolyee_id, empolyee_name,
count(chemical_name)
FROM (manage_project)
GROUP BY empolyee_id;

select * from project_count;

CREATE VIEW superviser_name AS
SELECT a.empolyee_id, a.first_name, a.last_name, a.job,
concat(b.first_name, " ", b.last_name) AS "Reports_to"
FROM empolyees AS a
LEFT JOIN empolyees AS b
ON a.superviser = b.Empolyee_id;

CREATE VIEW supervised_count AS
SELECT reports_to, count(empolyee_id)
from superviser_name
group by reports_to;

SELECT * FROM superviser_name;
SELECT * FROM supervised_count;

CREATE VIEW hourly_pay_statistic AS
SELECT job, 
min(hourly_pay) AS MIN,
avg(hourly_pay) AS AVG,
max(hourly_pay) AS MAX,
sum(hourly_pay) AS SUM
FROM empolyees
Group by job;

select * from hourly_pay_statistic;

CREATE VIEW profit AS
SELECT sum(a.amount) AS total_expense,
       sum(b.amount) AS total_income, 
       (sum(a.amount) * -1) + sum(b.amount) AS profit
FROM expense AS a
INNER JOIN income AS b;

SELECT * FROM profit;

-- Stored Procedures
delimiter //
CREATE PROCEDURE update_sales (IN id INT, IN new_sold INT) -- input (drugid, sold) data to update
BEGIN
UPDATE sales SET sold = new_sold WHERE drugid=id;
end; //
delimiter ;

call update_sales (1, 58630);


delimiter // 
CREATE PROCEDURE insert_infile_not_exist (IN tb_name VARCHAR(255), 
                                   IN id_col VARCHAR(255),
						           IN file_path VARCHAR(255)) -- input ('table_name', 'id_column_name', 'file_path') -- 1064 syntax error
BEGIN
SET GLOBAL LOCAL_INFILE=TRUE;

SET @sql1 = concat('CREATE TEMPORARY TABLE temp LIKE ', tb_name);
PREPARE stmt1 FROM @sql1;
EXECUTE stmt1;
DEALLOCATE PREPARE stmt1;

SET @sql2 = concat('LOAD DATA LOCAL INFILE ', file_path,  'INTO TABLE temp FIELDS TERMINATED BY \',\' ENCLOSED BY \'"\' LINES TERMINATED BY \'\n\' IGNORE 1 LINES');
PREPARE stmt2 FROM @sql2;
EXECUTE stmt2;
DEALLOCATE PREPARE stmt2;
    
SET @sql3 = concat('INSERT INTO ', tb_name, 'SELECT * FROM temp WHERE NOT EXISTS(SELECT * FROM',  tb_name, 'WHERE (', tb_name.id_col, ' = ', temp.id_col, '))');     
PREPARE stmt3 FROM @sql3;
EXECUTE stmt3;
DEALLOCATE PREPARE stmt3;

DROP TEMPORARY TABLE temp;
END;//
delimiter ;
drop PROCEDURE insert_infile_not_exist;
call insert_infile_not_exist('drug_info', 'drugid', 'D:/Work repository/NN_test_build/MySQL/full_drug_list.csv');
