import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function CustomerProfileForm() {
  const [values, setValues] = useState({});
  const [submitted, setSubmitted] = useState(false);

  const errors = useMemo(() => {
    return Object.fromEntries(
      props.fields.flatMap((field) => {
        const value = values[field.id];
        if (value === undefined || value === "") {
          return [[field.id, "Required"]];
        }
        if (field.type === "number") {
          const number = Number(value);
          if (!Number.isFinite(number)) {
            return [[field.id, "Enter a valid number"]];
          }
          if (number < field.minimum || number > field.maximum) {
            return [[field.id, `Use a value from ${field.minimum} to ${field.maximum}`]];
          }
        }
        return [];
      }),
    );
  }, [values]);

  const updateValue = (id, value) => {
    setValues((current) => ({ ...current, [id]: value }));
  };

  const handleSubmit = () => {
    if (Object.keys(errors).length > 0) {
      return;
    }
    setSubmitted(true);
    submitElement(values);
  };

  return (
    <Card className="mba-profile-card mb-24 mt-4 w-full">
      <CardHeader className="pb-6">
        <CardTitle>Customer profile</CardTitle>
        <CardDescription>
          Fill in the customer attributes used by the five churn models.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="grid"
          style={{
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            columnGap: "2rem",
            rowGap: "1.5rem",
          }}
        >
          {props.fields.map((field) => (
            <div
              key={field.id}
              className="grid content-start"
              style={{ rowGap: "0.625rem" }}
            >
              <Label className="block leading-5" htmlFor={field.id}>
                {field.label}
              </Label>
            {field.type === "select" ? (
              <Select
                disabled={submitted}
                value={values[field.id] || ""}
                onValueChange={(value) => updateValue(field.id, value)}
              >
                <SelectTrigger id={field.id}>
                  <SelectValue placeholder="Select an option" />
                </SelectTrigger>
                <SelectContent>
                  {field.options.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                id={field.id}
                type="number"
                min={field.minimum}
                max={field.maximum}
                step={field.step}
                disabled={submitted}
                value={values[field.id] || ""}
                placeholder={`${field.minimum}–${field.maximum}`}
                onChange={(event) => updateValue(field.id, event.target.value)}
              />
            )}
              {errors[field.id] && values[field.id] !== undefined && (
                <span className="min-h-4 text-xs text-destructive">
                  {errors[field.id]}
                </span>
              )}
            </div>
          ))}
        </div>
      </CardContent>
      <CardFooter className="mt-8 flex items-center justify-between gap-4 pb-8">
        <Button variant="outline" disabled={submitted} onClick={() => cancelElement()}>
          Cancel
        </Button>
        <Button
          disabled={submitted || Object.keys(errors).length > 0}
          onClick={handleSubmit}
        >
          {submitted ? "Analyzing..." : "Analyze customer"}
        </Button>
      </CardFooter>
    </Card>
  );
}
